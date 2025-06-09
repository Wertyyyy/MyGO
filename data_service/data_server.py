import logging
import time
import asyncio
from typing import List
import contextlib
import importlib.util
import copy
import traceback
from functools import partial

import torch
import fire
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from vllm_service.vllm_client import VLLMClient
from tf_service.tf_client import TFClient
from data_service.typing.grpo_data import GRPOData, BatchedGRPOData
from data_service.typing.message import Conversation
from data_service.grouping import adaptive_grouping
from config.utils import ConfigManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class StepUpdateRequest(BaseModel):
    step: int


class DataRequest(BaseModel):
    step: int
    rank: int
    update_step: bool


class DataResponse(BaseModel):
    data: List[BatchedGRPOData]


class DataServer:
    def __init__(
        self,
        config: ConfigManager,
    ):
        self.config = config
        if self.config.data_server.use_ref == "auto":
            self.use_ref = self.config.training.grpo_beta > 0
        elif self.config.data_server.use_ref == "always":
            self.use_ref = True
        elif self.config.data_server.use_ref == "never":
            self.use_ref = False
        else:
            raise ValueError(
                f"Invalid use_ref value: {self.config.data_server.use_ref}"
            )

        logger.info(f"Using reference model: {self.use_ref}")

        self.current_step = 0
        self.pregenerate_steps = self.config.data_server.pregenerate_steps
        self.fetching_tasks = {}

        self._initialize_components()

    def _initialize_components(self):
        # Load dataset
        dataset_module = importlib.import_module(self.config.dataset.train_impl_path)
        self.dataset = dataset_module.TrainingDatasetImpl(
            dataset_path=self.config.dataset.train_dataset_path,
            system_prompt_path=self.config.dataset.system_prompt_path,
            template_path=self.config.dataset.template_path,
        )
        logger.info(
            f"Loaded dataset implementation: {self.config.dataset.train_impl_path}"
        )

        # Load processor
        processor_module = importlib.import_module(self.config.processor.impl_path)
        self.processor = processor_module.TFProcessorImpl(
            init_params=self.config.processor.init_params.to_dict()
        )
        logger.info(
            f"Loaded processor implementation: {self.config.processor.impl_path}"
        )

        # Load reward functions
        self.reward_fns = {}
        for reward_name, reward_path in self.config.reward.to_dict().items():
            reward_module = importlib.import_module(reward_path)
            self.reward_fns[reward_name] = reward_module.reward
            logger.info(f"Loaded reward function: {reward_name} from {reward_path}")

        self._filter_dataset()
        self._create_dataloader()

    def _filter_dataset(self):
        def filter_fn(example, collate_fn):
            conversations, _ = collate_fn([example])
            seq_len = self.processor.get_seq_length(conversations[0])
            return seq_len <= self.config.length.max_prompt_length

        original_size = len(self.dataset.dataset)
        logger.info(f"Training dataset size before filtering: {original_size}")
        self.dataset.dataset = self.dataset.dataset.filter(
            partial(filter_fn, collate_fn=self.dataset.collate_fn),
            batched=False,
        )
        filtered_size = len(self.dataset.dataset)
        logger.info(f"Training dataset size after filtering: {filtered_size}")

    def _create_dataloader(self):
        def collate_wrapper(examples):
            return self.dataset.collate_fn(examples)

        dataloader = torch.utils.data.DataLoader(
            self.dataset.dataset,
            batch_size=self.config.data_server.global_batch_size,
            shuffle=True,
            collate_fn=collate_wrapper,
            drop_last=True,
        )
        self.dataloader = iter(dataloader)
        logger.info(
            f"DataLoader created with batch size: {self.config.data_server.global_batch_size}"
        )

    async def initialize_clients(self):
        self.policy_model_vllm_client = VLLMClient(
            host=self.config.network.vllm_host,
            server_port=self.config.network.vllm_port,
        )
        clients_to_init = [self.policy_model_vllm_client.initialize()]
        if self.use_ref:
            self.ref_model_tf_client = TFClient(
                host=self.config.network.tf_host, port=self.config.network.tf_port
            )
            clients_to_init.append(self.ref_model_tf_client.initialize())
        else:
            self.ref_model_tf_client = None

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
            n=self.config.data_server.per_prompt_generation_count,
            max_tokens=self.config.length.max_response_length,
            temperature=self.config.data_server.temperature,
        )

        conversations: List[Conversation] = []
        batched_prompt_token_ids = []
        batched_response_token_ids = []
        for rollout in vllm_outputs["completions"]:
            conversation_ = copy.deepcopy(conversation)
            conversation_.add_message(role="assistant", content=rollout)
            conversations.append(conversation_)

            seq_length = self.processor.get_seq_length(conversation_)
            prompt_token_ids, response_token_ids = (
                self.processor.get_prompt_response_token_ids(conversation_)
            )
            assert prompt_token_ids == vllm_outputs["prompt_token_ids"]
            assert len(prompt_token_ids) + len(response_token_ids) == seq_length

            assert len(prompt_token_ids) <= self.config.length.max_prompt_length
            # However, the prompt and response token ids may be slightly longer than the max length
            # Hopefully this will never happen
            if (
                len(prompt_token_ids) + len(response_token_ids)
                > self.config.length.max_length
            ):
                response_token_ids = response_token_ids[
                    : self.config.length.max_length - len(prompt_token_ids)
                ]
                logger.warning(
                    f"Token ids are longer than the max length, truncating to {len(response_token_ids)}\n"
                    f"Conversation: {conversation_}\n"
                    f"Prompt token ids: {prompt_token_ids}\n"
                    f"Response token ids (before truncation): {self.processor.get_prompt_response_token_ids(conversation_)[1]}\n"
                    f"Response token ids (after truncation): {response_token_ids}"
                )

            batched_prompt_token_ids.append(prompt_token_ids)
            batched_response_token_ids.append(response_token_ids)

        # Only get logprobs if using reference model
        if self.use_ref:
            tf_results = await self.ref_model_tf_client.get_logprobs_and_input_ids(
                conversations
            )
            batched_ref_logprobs = tf_results["batched_logprobs"]
        else:
            # Create dummy logprobs and input_ids if not using ref model
            batched_ref_logprobs = [None for _ in conversations]

        group_resp_token_sum = sum(
            len(resp_ids) for resp_ids in batched_response_token_ids
        )

        grpo_data_list: List[GRPOData] = []
        for response_idx, (
            conversation,
            stop_reason,
            ref_logprobs,
            prompt_token_ids,
            response_token_ids,
        ) in enumerate(
            zip(
                conversations,
                vllm_outputs["finish_reasons"],
                batched_ref_logprobs,
                batched_prompt_token_ids,
                batched_response_token_ids,
            )
        ):
            if ref_logprobs is not None and len(ref_logprobs) != len(
                response_token_ids
            ):
                logger.warning(
                    f"ref_logprobs length {len(ref_logprobs)} != response_token_ids length {len(response_token_ids)}"
                )

            grpo_data_list.append(
                GRPOData(
                    prompt_idx=prompt_idx,
                    response_idx=response_idx,
                    conversation=conversation,
                    solution=solution,
                    stop_reason=stop_reason,
                    prompt_token_ids=prompt_token_ids,
                    response_token_ids=response_token_ids,
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

        all_data = adaptive_grouping(
            all_data,
            gpu_num=self.config.data_server.gpu_num,
            token_budget=self.config.data_server.token_budget,
            max_micro_step_num=self.config.data_server.max_micro_step_num,
        )

        return all_data

    async def update_step(self, step: int):
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

    async def get_data(self, step: int, rank: int) -> List[BatchedGRPOData]:
        if step not in self.fetching_tasks:
            raise HTTPException(status_code=404, detail="Data not found")
        global_step_data = await self.fetching_tasks[step]
        rank_data = []
        for micro_step_data in global_step_data.data:
            rank_data.append(micro_step_data.data[rank])
        return rank_data

    async def close(self):
        if self.policy_model_vllm_client:
            await self.policy_model_vllm_client.close()
        if self.ref_model_tf_client:
            await self.ref_model_tf_client.close()

    async def reset(self):
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
        await self.update_step(0)

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
            if request.update_step:
                await server.update_step(request.step)

            data = await server.get_data(request.step, request.rank)

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
    config = ConfigManager(config_file)
    server = DataServer(config=config)
    app = create_app(server)

    uvicorn_config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=config.network.data_port,
        timeout_keep_alive=7200,
    )
    server_instance = uvicorn.Server(uvicorn_config)
    asyncio.run(server_instance.serve())


if __name__ == "__main__":
    fire.Fire(main)
