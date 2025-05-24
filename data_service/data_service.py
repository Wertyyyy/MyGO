from typing import List, Dict, Any
import logging
import asyncio
import importlib
import copy
from dataclasses import dataclass
import threading

import numpy as np
import torch

from vllm_service.vllm_client import VLLMClient
from tf_service.tf_client import TFClient
from data_service.typing.grpo_data import GRPOData
from data_service.typing.messages import Conversation
from data_service.grouping import adaptive_grouping
from utils import Metrics

logger = logging.getLogger(__name__)


@dataclass
class DataService:
    dataset_impl: Dict[str, Any]
    processor_impl: Dict[str, Any]
    reward_impl: Dict[str, Any]

    gpu_num: int
    token_budget: int
    global_batch_size: int
    per_prompt_generation_count: int
    max_response_length: int
    max_prompt_length: int
    max_micro_step_num: int
    temperature: float

    vllm_host: str
    vllm_server_port: int
    vllm_nccl_port: int
    tf_host: str
    tf_port: int

    def _filter_dataset(self):
        def filter_fn(example):
            conversations, _ = self.dataset.collate_fn([example])
            seq_len = self.processor.get_seq_length(conversations[0])
            return seq_len <= self.max_prompt_length

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
            batch_size=self.global_batch_size,
            shuffle=True,
            collate_fn=collate_wrapper,
            drop_last=True,
        )
        self.dataloader = iter(dataloader)
        logger.info(f"DataLoader created with batch size: {self.global_batch_size}")

    def _create_event_loop(self):
        def run_event_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        self.loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(
            target=run_event_loop, args=(self.loop,), daemon=True
        )
        loop_thread.start()
        logger.info("Event loop created and started")

    def _run_async_in_loop(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def _init_clients(self):
        async def _init_clients_async():
            self.policy_model_vllm_client = VLLMClient(
                host=self.vllm_host,
                server_port=self.vllm_server_port,
                nccl_port=self.vllm_nccl_port,
            )
            self.ref_model_tf_client = TFClient(host=self.tf_host, port=self.tf_port)
            clients_to_init = [
                self.policy_model_vllm_client.initialize(),
                self.ref_model_tf_client.initialize(),
            ]

            await asyncio.gather(*clients_to_init)
            logger.info("Data service clients initialized")

        return self._run_async_in_loop(_init_clients_async()).result()

    def _init_nccl(self):
        async def _init_nccl_async():
            if self.policy_model_vllm_client:
                await self.policy_model_vllm_client.init_nccl()
            else:
                raise ValueError("Policy model VLLM client not initialized")

        return self._run_async_in_loop(_init_nccl_async()).result()

    def __post_init__(self):
        dataset_module = importlib.import_module(self.dataset_impl["path"])
        self.dataset = dataset_module.DatasetImpl(**self.dataset_impl["params"])
        logger.info(f"Loaded dataset implementation: {self.dataset_impl['path']}")

        processor_module = importlib.import_module(self.processor_impl["path"])
        self.processor = processor_module.DataServerProcessorImpl(
            **self.processor_impl["params"]
        )
        logger.info(f"Loaded processor implementation: {self.processor_impl['path']}")

        self.reward_fns = {}
        for reward_name, reward_spec in self.reward_impl.items():
            reward_module = importlib.import_module(reward_spec["path"])
            self.reward_fns[reward_name] = reward_module.reward
            logger.info(
                f"Loaded reward function: {reward_name} from {reward_spec['path']}"
            )

        self._filter_dataset()
        self._create_dataloader()
        self._create_event_loop()
        self._init_clients()
        self._init_nccl()

    def get_batch(self):
        try:
            conversations, solutions = next(self.dataloader)
        except StopIteration:
            logger.info("Reloading dataset")
            self._create_dataloader()
            conversations, solutions = next(self.dataloader)

        return conversations, solutions

    async def _fetch_data_per_prompt(
        self, conversation: Conversation, solution: str, prompt_idx: int
    ) -> List[GRPOData]:
        # Generate responses using VLLM
        vllm_outputs = await self.policy_model_vllm_client.generate(
            conversation=conversation,
            n=self.per_prompt_generation_count,
            max_tokens=self.max_response_length,
            temperature=self.temperature,
        )
        rollouts = vllm_outputs["completions"]

        # Create messages with completions
        conversations = []
        seq_lengths = []
        for rollout in rollouts:
            conversation_with_completion = copy.deepcopy(conversation)
            conversation_with_completion.append(
                {"role": "assistant", "content": rollout}
            )
            conversations.append(conversation_with_completion)
            seq_lengths.append(
                self.processor.get_seq_length(conversation_with_completion)
            )

        # Get reference logprobs using TF client
        tf_results = await self.ref_model_tf_client.get_logprobs(conversations)
        batched_ref_logprobs = tf_results["batched_logprobs"]

        group_resp_token_num = sum(len(logprobs) for logprobs in batched_ref_logprobs)

        grpo_data_list = []
        for response_idx, (
            conversation,
            length,
            stop_reason,
            ref_logprobs,
        ) in enumerate(
            zip(
                conversations,
                seq_lengths,
                vllm_outputs["finish_reasons"],
                batched_ref_logprobs,
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
                    ref_logprobs=ref_logprobs,
                    group_resp_token_num=group_resp_token_num,
                    group_seq_num=len(conversations),
                )
            )

        return grpo_data_list

    def fetch_data(self):
        async def _fetch_data():
            logger.debug("Fetching data")

            conversations, solutions = self.get_batch()

            tasks = []
            for idx, (conversation, solution) in enumerate(
                zip(conversations, solutions)
            ):
                tasks.append(self._fetch_data_per_prompt(conversation, solution, idx))

            all_data: List[List[GRPOData]] = await asyncio.gather(*tasks)
            return all_data

        # NOTE: This is a workaround to calculate rewards synchronously in the main thread.
        class FetchDataFuture:
            def __init__(self, data_service, future):
                self.data_service = data_service
                self.future = future

            def result(self, metrics: Metrics):
                all_data: List[List[GRPOData]] = self.future.result()
                for grpo_data_list in all_data:
                    self.data_service._calculate_rewards(grpo_data_list)
                self.data_service._statistics(all_data, metrics)

                global_step_data = adaptive_grouping(
                    all_data,
                    self.data_service.gpu_num,
                    self.data_service.token_budget,
                    self.data_service.max_micro_step_num,
                    metrics,
                )
                return global_step_data

        return FetchDataFuture(self, self._run_async_in_loop(_fetch_data()))

    def update_vllm_weights(self, state_dict):
        async def _update_vllm_weights():
            if self.policy_model_vllm_client:
                await self.policy_model_vllm_client.update_weights_nccl(state_dict)
            else:
                raise ValueError("Policy model VLLM client not initialized")

        return self._run_async_in_loop(_update_vllm_weights()).result()

    def _calculate_rewards(self, grpo_data_list: List[GRPOData]) -> List[GRPOData]:
        total_rewards = []
        for data in grpo_data_list:
            rewards = [
                reward_fn(data.conversation[-1]["content"], data.solution)
                for reward_fn in self.reward_fns.values()
            ]
            data.rewards = rewards
            total_rewards.append(sum(rewards))

        # Calculate advantages
        mean_reward = torch.tensor(total_rewards).mean()
        std_reward = torch.tensor(total_rewards).std()
        advantages = (
            (torch.tensor(total_rewards) - mean_reward) / (std_reward + 1e-4)
        ).view(-1)

        for data, advantage in zip(grpo_data_list, advantages):
            data.advantage = advantage.item()

        return grpo_data_list

    def _statistics(self, all_data: List[List[GRPOData]], metrics: Metrics):
        flatten_data: List[GRPOData] = []
        for grouped_data in all_data:
            flatten_data.extend(grouped_data)

        logger.debug("===== Data Statistics =====")
        logger.debug(f"Total samples: {len(flatten_data)}")

        # Length statistics
        lengths = np.array([data.length for data in flatten_data])
        prompt_lengths = np.array([data.prompt_length for data in flatten_data])
        response_lengths = np.array([data.response_length for data in flatten_data])

        logger.debug("Length Statistics:")
        logger.debug(
            f"  Total length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}, "
            f"max: {np.max(lengths):.2f}, min: {np.min(lengths):.2f}"
        )
        logger.debug(
            f"  Prompt length: {np.mean(prompt_lengths):.2f} ± {np.std(prompt_lengths):.2f}, "
            f"max: {np.max(prompt_lengths):.2f}, min: {np.min(prompt_lengths):.2f}"
        )
        logger.debug(
            f"  Response length: {np.mean(response_lengths):.2f} ± {np.std(response_lengths):.2f}, "
            f"max: {np.max(response_lengths):.2f}, min: {np.min(response_lengths):.2f}"
        )

        metrics.add("Data/length/total/mean", np.mean(lengths).item())
        metrics.add("Data/length/total/max", np.max(lengths).item())
        metrics.add("Data/length/total/min", np.min(lengths).item())
        metrics.add("Data/length/prompt/mean", np.mean(prompt_lengths).item())
        metrics.add("Data/length/prompt/max", np.max(prompt_lengths).item())
        metrics.add("Data/length/prompt/min", np.min(prompt_lengths).item())
        metrics.add("Data/length/response/mean", np.mean(response_lengths).item())
        metrics.add("Data/length/response/max", np.max(response_lengths).item())
        metrics.add("Data/length/response/min", np.min(response_lengths).item())

        # Finish reason statistics
        finish_reasons = [data.stop_reason for data in flatten_data]
        by_stop = finish_reasons.count("stop")
        by_length = finish_reasons.count("length")

        logger.debug("Finish Reason Statistics:")
        logger.debug(f"  stop: {by_stop} ({by_stop / len(finish_reasons) * 100:.1f}%)")
        logger.debug(
            f"  length: {by_length} ({by_length / len(finish_reasons) * 100:.1f}%)"
        )

        metrics.add("Data/finish_reason/length", by_length / len(finish_reasons))

        # Reward statistics
        reward_names = [fn.__name__ for fn in self.reward_fns.values()]

        logger.debug("Reward Statistics:")
        all_rewards = []
        for data in flatten_data:
            if data.rewards is not None:
                all_rewards.append(data.rewards)

        if all_rewards:
            all_rewards = np.array(all_rewards)
            for rwd_idx, rwd_name in enumerate(reward_names):
                rwd_i = all_rewards[:, rwd_idx]
                logger.debug(
                    f"  {rwd_name}: {np.mean(rwd_i):.4f} ± {np.std(rwd_i):.4f}"
                )
                metrics.add(f"Reward/{rwd_name}/mean", np.mean(rwd_i).item())
                metrics.add(f"Reward/{rwd_name}/std", np.std(rwd_i).item())

            total_rewards = np.sum(all_rewards, axis=1)
            logger.debug(
                f"  Total reward: {np.mean(total_rewards):.4f} ± {np.std(total_rewards):.4f}"
            )
            metrics.add("Reward/total/mean", np.mean(total_rewards).item())
            metrics.add("Reward/total/std", np.std(total_rewards).item())
        else:
            logger.debug("No reward data available for statistics")

        logger.debug("===========================")

    def close(self):
        async def _close():
            if self.policy_model_vllm_client:
                await self.policy_model_vllm_client.close()
            if self.ref_model_tf_client:
                await self.ref_model_tf_client.close()

        return self._run_async_in_loop(_close()).result()
