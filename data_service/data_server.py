import logging
import time
import asyncio
from typing import List, Any, Dict
import contextlib
import importlib.util
import copy
from dataclasses import dataclass
import traceback

import torch
import fire
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from vllm_service.vllm_client import VLLMClient
from tf_service.tf_client import TFClient
from data_service.typing.grpo_data import GRPOData
from data_service.typing.message import Conversation

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class StepUpdateRequest(BaseModel):
    step: int


class DataRequest(BaseModel):
    step: int


class DataResponse(BaseModel):
    data: List[List[GRPOData]]


@dataclass
class GenerationConfig:
    global_batch_size: int
    per_prompt_generation_count: int
    max_response_length: int
    max_prompt_length: int
    temperature: float
    pregenerate_steps: int


@dataclass
class NetworkConfig:
    vllm_host: str
    vllm_port: int
    tf_host: str
    tf_port: int


class DataServer:
    def __init__(
        self,
        dataset_impl: Dict[str, Any],
        processor_impl: Dict[str, Any],
        reward_impl: Dict[str, Any],
        generation_config: GenerationConfig,
        network_config: NetworkConfig,
    ):
        self.dataset_impl = dataset_impl
        self.processor_impl = processor_impl
        self.reward_impl = reward_impl
        self.generation_config = generation_config
        self.network_config = network_config

        self.current_step = 0
        self.pregenerate_steps = self.generation_config.pregenerate_steps
        self.fetching_tasks = {}

        self._initialize_components()

    def _initialize_components(self):
        """Initialize dataset, processor, and reward functions"""
        # Load dataset
        dataset_module = importlib.import_module(self.dataset_impl["path"])
        self.dataset = dataset_module.DatasetImpl(**self.dataset_impl["params"])
        logger.info(f"Loaded dataset implementation: {self.dataset_impl['path']}")

        # Load processor
        processor_module = importlib.import_module(self.processor_impl["path"])
        self.processor = processor_module.TFProcessorImpl(
            **self.processor_impl["params"]
        )
        logger.info(f"Loaded processor implementation: {self.processor_impl['path']}")

        # Load reward functions
        self.reward_fns = {}
        for reward_name, reward_spec in self.reward_impl.items():
            reward_module = importlib.import_module(reward_spec["path"])
            self.reward_fns[reward_name] = reward_module.reward
            logger.info(
                f"Loaded reward function: {reward_name} from {reward_spec['path']}"
            )

        self._filter_dataset()
        self._create_dataloader()

    def _filter_dataset(self):
        def filter_fn(example):
            conversations, _ = self.dataset.collate_fn([example])
            seq_len = self.processor.get_seq_length(conversations[0])
            return seq_len <= self.generation_config.max_prompt_length

        original_size = len(self.dataset.dataset)
        logger.info(f"Dataset size before filtering: {original_size}")
        self.dataset.dataset = self.dataset.dataset.filter(filter_fn, batched=False)
        filtered_size = len(self.dataset.dataset)
        logger.info(f"Dataset size after filtering: {filtered_size}")

    def _create_dataloader(self):
        def collate_wrapper(examples):
            return self.dataset.collate_fn(examples)

        dataloader = torch.utils.data.DataLoader(
            self.dataset.dataset,
            batch_size=self.generation_config.global_batch_size,
            shuffle=True,
            collate_fn=collate_wrapper,
            drop_last=True,
        )
        self.dataloader = iter(dataloader)
        logger.info(
            f"DataLoader created with batch size: {self.generation_config.global_batch_size}"
        )

    async def initialize_clients(self):
        """Initialize VLLM and TF clients"""
        self.policy_model_vllm_client = VLLMClient(
            host=self.network_config.vllm_host,
            server_port=self.network_config.vllm_port,
        )
        self.ref_model_tf_client = TFClient(
            host=self.network_config.tf_host, port=self.network_config.tf_port
        )

        clients_to_init = [
            self.policy_model_vllm_client.initialize(),
            self.ref_model_tf_client.initialize(),
        ]

        await asyncio.gather(*clients_to_init)
        logger.info("Data service clients initialized")

    def _get_batch(self):
        """Get next batch from dataloader"""
        try:
            conversations, solutions = next(self.dataloader)
            return conversations, solutions
        except StopIteration:
            # Recreate dataloader when exhausted
            self._create_dataloader()
            conversations, solutions = next(self.dataloader)
            return conversations, solutions

    def _calculate_rewards(self, grpo_data_list: List[GRPOData]) -> List[GRPOData]:
        total_rewards = []
        for data in grpo_data_list:
            # Get the content of the last message (assistant's response)
            last_message = data.conversation.messages[-1]
            assistant_response = last_message.content

            rewards = {
                reward_name: reward_fn(assistant_response, data.solution)
                for reward_name, reward_fn in self.reward_fns.items()
            }
            data.rewards = rewards
            total_rewards.append(data.reward_sum)

        # Calculate advantages
        mean_reward = torch.tensor(total_rewards).mean()
        std_reward = torch.tensor(total_rewards).std()
        advantages = (
            (torch.tensor(total_rewards) - mean_reward) / (std_reward + 1e-4)
        ).view(-1)

        for data, advantage in zip(grpo_data_list, advantages):
            data.advantage = advantage.item()

        return grpo_data_list

    async def _fetch_data_per_prompt(
        self, conversation: Conversation, solution: str, prompt_idx: int
    ) -> List[GRPOData]:
        vllm_outputs = await self.policy_model_vllm_client.generate(
            conversation=conversation,
            n=self.generation_config.per_prompt_generation_count,
            max_tokens=self.generation_config.max_response_length,
            temperature=self.generation_config.temperature,
        )

        conversations: List[Conversation] = []
        seq_lengths = []
        for rollout in vllm_outputs["completions"]:
            conversation_ = copy.deepcopy(conversation)
            conversation_.add_message(role="assistant", content=rollout)
            conversations.append(conversation_)
            seq_lengths.append(self.processor.get_seq_length(conversation_))

        tf_results = await self.ref_model_tf_client.get_logprobs_and_input_ids(
            conversations
        )
        batched_ref_logprobs = tf_results["batched_logprobs"]
        batched_input_ids = tf_results["batched_input_ids"]
        group_resp_token_sum = sum(len(logprobs) for logprobs in batched_ref_logprobs)

        grpo_data_list: List[GRPOData] = []
        for response_idx, (
            conversation,
            length,
            stop_reason,
            ref_logprobs,
            input_ids,
        ) in enumerate(
            zip(
                conversations,
                seq_lengths,
                vllm_outputs["finish_reasons"],
                batched_ref_logprobs,
                batched_input_ids,
            )
        ):
            grpo_data_list.append(
                GRPOData(
                    prompt_idx=prompt_idx,
                    response_idx=response_idx,
                    conversation=conversation,
                    solution=solution,
                    stop_reason=stop_reason,
                    length=length,
                    resp_token_ids=input_ids,
                    ref_logprobs=ref_logprobs,
                    group_resp_token_sum=group_resp_token_sum,
                    group_seq_num=len(conversations),
                )
            )

        grpo_data_list = self._calculate_rewards(grpo_data_list)
        return grpo_data_list

    async def _fetch_data(self) -> List[List[GRPOData]]:
        """Generate data for all ranks in a step"""
        conversations, solutions = self._get_batch()

        tasks = []
        for idx, (conversation, solution) in enumerate(zip(conversations, solutions)):
            tasks.append(self._fetch_data_per_prompt(conversation, solution, idx))

        all_data: List[List[GRPOData]] = await asyncio.gather(*tasks)

        global_resp_token_sum = 0
        global_seq_num = 0
        global_group_num = 0
        global_resp_max_len = 0
        for grouped_data in all_data:
            for data in grouped_data:
                global_resp_token_sum += data.response_length
                global_resp_max_len = max(global_resp_max_len, data.response_length)
                global_seq_num += 1
            global_group_num += 1

        for grouped_data in all_data:
            for data in grouped_data:
                data.global_resp_token_sum = global_resp_token_sum
                data.global_seq_num = global_seq_num
                data.global_group_num = global_group_num
                data.global_resp_max_len = global_resp_max_len

        return all_data

    async def update_step(self, step: int):
        """Update current step and pregenerate data for future steps"""
        logger.info(f"Updating step from {self.current_step} to {step}")
        self.current_step = step

        steps_to_remove = [s for s in self.fetching_tasks.keys() if s < step]
        for old_step in steps_to_remove:
            del self.fetching_tasks[old_step]
            logger.debug(f"Removed fetching task for old step {old_step}")

        for step_idx in range(step, step + self.pregenerate_steps + 1):
            if step_idx not in self.fetching_tasks:
                logger.info(f"Fetching data for step {step_idx}")
                self.fetching_tasks[step_idx] = asyncio.create_task(self._fetch_data())

        logger.info(f"Step update completed. Current step: {self.current_step}")

    async def get_data(self, step: int) -> List[List[GRPOData]]:
        """Get data for a specific step"""
        if step not in self.fetching_tasks:
            raise HTTPException(status_code=404, detail="Data not found")
        return await self.fetching_tasks[step]

    async def close(self):
        if self.policy_model_vllm_client:
            await self.policy_model_vllm_client.close()
        if self.ref_model_tf_client:
            await self.ref_model_tf_client.close()

    async def reset(self):
        """Reset the server state to initial conditions"""
        logger.info("Resetting data server state")

        # Cancel all pending fetching tasks
        for step, task in self.fetching_tasks.items():
            if not task.done():
                task.cancel()
                logger.debug(f"Cancelled fetching task for step {step}")

        # Clear fetching tasks
        self.fetching_tasks.clear()

        # Reset current step
        self.current_step = 0

        # Recreate dataloader to get fresh data
        self._create_dataloader()

        logger.info("Data server state reset completed")


def create_app(server: DataServer):
    app = FastAPI()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        await server.initialize_clients()
        yield
        await server.close()

    app.router.lifespan_context = lifespan

    @app.middleware("http")
    async def log_requests(request, call_next):
        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"

        logger.info(
            f"Received request: {request.method} {request.url.path} from {client_ip}"
        )
        logger.debug(f"Request headers: {dict(request.headers)}")

        response = await call_next(request)

        process_time = time.time() - start_time
        logger.info(
            f"Request completed: {request.method} {request.url.path} "
            f"- Status: {response.status_code} - Time: {process_time:.2f}s"
        )

        return response

    @app.get("/health/")
    async def health():
        return {"status": "ok"}

    @app.post("/update_step/")
    async def update_step(request: StepUpdateRequest):
        """Update current step and pregenerate data for future steps"""
        try:
            await server.update_step(request.step)
            return {"status": "success", "current_step": server.current_step}
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(f"Error updating step: {e}\nTraceback:\n{tb_str}")
            raise HTTPException(status_code=500, detail=f"{e}\nTraceback:\n{tb_str}")

    @app.post("/reset/")
    async def reset():
        """Reset server state to initial conditions"""
        try:
            await server.reset()
            return {"status": "success", "message": "Server state reset successfully"}
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(f"Error resetting server: {e}\nTraceback:\n{tb_str}")
            raise HTTPException(status_code=500, detail=f"{e}\nTraceback:\n{tb_str}")

    @app.post("/fetch_data/", response_model=DataResponse)
    async def fetch_data(request: DataRequest):
        """Fetch training data for specific step"""
        try:
            start_time = time.time()

            data = await server.get_data(request.step)

            end_time = time.time()
            logger.info(
                f"Data fetch for step {request.step} completed in {end_time - start_time:.2f} seconds"
            )

            return DataResponse(data=data)

        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(
                f"Error fetching data for step {request.step}: {e}\n"
                f"Traceback:\n{tb_str}"
            )
            raise HTTPException(status_code=500, detail=f"{e}\nTraceback:\n{tb_str}")

    return app


def main(config_file: str):
    # Load the Python config module
    spec = importlib.util.spec_from_file_location("config_module", config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    # Extract data_server configuration
    data_config = config_module.data_server
    processor_config = config_module.processor

    # Create configuration objects from nested config
    generation_config = GenerationConfig(**data_config["generation_config"])
    network_config = NetworkConfig(**data_config["network_config"])

    # Prepare dataset implementation config
    dataset_impl = {
        "path": data_config["dataset"]["impl_path"],
        "params": data_config["dataset"]["params"],
    }

    # Prepare processor implementation config
    processor_impl = {
        "path": processor_config["impl_path"],
        "params": processor_config["params"],
    }

    # Prepare reward implementation config
    reward_impl = {}
    for reward_name, reward_path in data_config["reward"].items():
        reward_impl[reward_name] = {"path": reward_path}

    server = DataServer(
        dataset_impl=dataset_impl,
        processor_impl=processor_impl,
        reward_impl=reward_impl,
        generation_config=generation_config,
        network_config=network_config,
    )

    app = create_app(server)

    config_obj = uvicorn.Config(
        app,
        host=data_config["host"],
        port=data_config["port"],
        timeout_keep_alive=7200,
        log_level="debug",
        access_log=True,
    )
    server_instance = uvicorn.Server(config_obj)
    asyncio.run(server_instance.serve())


if __name__ == "__main__":
    fire.Fire(main)
