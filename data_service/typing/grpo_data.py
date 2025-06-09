from typing import List, Optional, Union, Tuple, Dict, Any
import logging
import json
from datetime import datetime
import statistics

import torch
from pydantic import BaseModel, Field, field_validator, field_serializer

from data_service.typing.message import Conversation

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GRPOData(BaseModel):
    prompt_idx: Union[str, int] = Field(...)
    response_idx: int = Field(...)

    conversation: Conversation = Field(...)
    solution: str = Field(...)
    stop_reason: str = Field(...)

    response_token_ids: List[int] = Field(...)
    prompt_token_ids: List[int] = Field(...)

    ref_logprobs: Optional[torch.Tensor] = Field(None)
    pol_logprobs: Optional[torch.Tensor] = Field(None)
    per_token_entropy: Optional[torch.Tensor] = Field(None)
    per_token_kl: Optional[torch.Tensor] = Field(None)
    per_token_loss: Optional[torch.Tensor] = Field(None)

    group_resp_token_sum: Optional[int] = Field(None)
    group_seq_num: Optional[int] = Field(None)
    global_resp_token_sum: Optional[int] = Field(None)
    global_seq_num: Optional[int] = Field(None)
    global_group_num: Optional[int] = Field(None)
    global_resp_max_len: Optional[int] = Field(None)

    rewards: Optional[Dict[str, float]] = Field(default_factory=dict)
    advantage: Optional[float] = Field(None)

    class Config:
        arbitrary_types_allowed = True

    @field_validator(
        "ref_logprobs",
        "pol_logprobs",
        "per_token_entropy",
        "per_token_kl",
        "per_token_loss",
        mode="before",
    )
    @classmethod
    def validate_tensor_fields(cls, v):
        if v is None:
            return v
        elif isinstance(v, torch.Tensor):
            if v.dim() != 1:
                raise ValueError(f"Field must be 1D tensor, got {v.dim()}D tensor")
            return v
        elif isinstance(v, list):
            if all(isinstance(x, (int, float)) for x in v):
                return torch.tensor(v, dtype=torch.float32, device=torch.device("cpu"))
            raise ValueError("List must contain only numbers, got mixed types")
        else:
            raise ValueError(
                f"Field must be None, List[float] or torch.Tensor, got {type(v)}"
            )

    @field_serializer(
        "ref_logprobs",
        "pol_logprobs",
        "per_token_entropy",
        "per_token_kl",
        "per_token_loss",
    )
    def serialize_tensor_fields(self, value):
        if value is None:
            return None
        elif isinstance(value, torch.Tensor):
            return value.cpu().tolist()
        return value

    @property
    def length(self) -> int:
        return len(self.response_token_ids) + len(self.prompt_token_ids)

    @property
    def response_length(self) -> int:
        return len(self.response_token_ids)

    @property
    def prompt_length(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def reward_sum(self) -> float:
        return sum(self.rewards.values())

    def log(self, prefix: str = "", idx: int = 0):
        logger.info(
            f"{prefix}Data {idx} (prompt_idx={self.prompt_idx}, response_idx={self.response_idx}, length={self.length}/{self.prompt_length}/{self.response_length}, stop_reason={self.stop_reason})"
        )


class BatchedGRPOData(BaseModel):
    token_budget: int = Field(..., description="Token budget for the batch")
    data: List[GRPOData] = Field(
        default_factory=list, description="List of GRPO data items"
    )

    @property
    def longest_seq_len(self) -> int:
        """The length of the longest sequence in the batch."""
        return max([item.length for item in self.data], default=0)

    @property
    def seq_num(self) -> int:
        """The number of sequences in the batch."""
        return len(self.data)

    @property
    def extra_seq_num(self) -> int:
        """The number of extra sequences in the batch."""
        return max(self.seq_num - 1, 0)

    @property
    def current_load(self) -> int:
        """The current load of the batch."""
        return self.longest_seq_len * self.seq_num

    @property
    def effective_load(self) -> int:
        """The effective load of the batch."""
        return sum([item.length for item in self.data])

    @property
    def efficiency(self) -> float:
        """The efficiency of the batch."""
        assert self.data, "Batch is empty"
        return self.effective_load / self.token_budget

    def fit_in(self, data: GRPOData, add_to_batch: bool = True) -> bool:
        """Fit an item into the batch."""
        if (
            max(self.longest_seq_len, data.length) * (self.seq_num + 1)
            > self.token_budget
        ):
            return False
        if add_to_batch:
            self.data.append(data)
            self.data.sort(key=lambda x: x.length, reverse=True)
        return True

    def pop(self) -> Optional[GRPOData]:
        assert self.seq_num > 0, "Batch is empty"
        return self.data.pop()

    def verify(self):
        """Verify the batch."""
        assert self.seq_num > 0, "Batch is empty"
        assert self.current_load <= self.token_budget, (
            f"Batch is over budget: {self.current_load} > {self.token_budget}"
        )

    def log(self, prefix: str = "", idx: int = 0):
        logger.info(
            f"{prefix}Batch {idx} (seq_num={self.seq_num}, token_budget={self.token_budget}, current_load={self.current_load}, efficiency={self.efficiency})"
        )
        for idx, item in enumerate(self.data):
            item.log(prefix + "    ", idx)

    def set_data_fields(self, field_name: str, value: List[Any]):
        for item, v in zip(self.data, value):
            setattr(item, field_name, v)

    def get_data_fields(self, field_name: str, device=None) -> List:
        if device is None:
            # FIXME: This is a hack to get the device of the current process
            from accelerate import PartialState

            state = PartialState()
            device = state.device

        result = []
        for item in self.data:
            value = getattr(item, field_name)
            if hasattr(value, "to") and hasattr(value, "device"):
                value = value.to(device)
            result.append(value)
        return result


class MicroStepGRPOData(BaseModel):
    gpu_num: int = Field(..., description="Number of GPUs")
    token_budget: int = Field(..., description="Token budget per batch")
    data: List[BatchedGRPOData] = Field(
        default_factory=list, description="List of batched GRPO data"
    )

    def model_post_init(self, __context):
        """Initialize empty batches to match GPU count."""
        while len(self.data) < self.gpu_num:
            self.data.append(BatchedGRPOData(token_budget=self.token_budget))

    def fit_in(self, data: GRPOData, add_to_batch: bool = True) -> bool:
        """Fit an item into the batch."""
        for batch in self.data:
            if batch.fit_in(data, add_to_batch):
                return True
        return False

    @property
    def total_seq_num(self) -> int:
        """The total number of sequences in the batch."""
        return sum(batch.seq_num for batch in self.data)

    @property
    def total_extra_seq_num(self) -> int:
        """The total number of extra sequences in the batch."""
        return sum(batch.extra_seq_num for batch in self.data)

    @property
    def seq_needed_num(self) -> int:
        if self.total_seq_num >= self.gpu_num:
            return 0
        return self.gpu_num - self.total_seq_num

    @property
    def total_effective_load(self) -> int:
        """The effective load of the batch."""
        return sum(batch.effective_load for batch in self.data)

    @property
    def max_load(self) -> int:
        """The max load of the batch."""
        return max(batch.current_load for batch in self.data)

    @property
    def efficiency(self) -> float:
        """The efficiency of the batch."""
        assert self.max_load > 0, "Max load should be greater than 0"
        return self.total_effective_load / (self.gpu_num * self.max_load)

    def pop(self) -> GRPOData:
        """Pop a GRPOData item from the batches, starting from the last batch."""
        for batch in reversed(self.data):
            if batch.seq_num > 0:
                return batch.pop()
        assert False

    def flatten(self) -> List[GRPOData]:
        """Get all GRPOData items from all batches."""
        result = []
        for batch in self.data:
            result.extend(batch.data)
        return result

    def verify(self):
        assert len(self.data) == self.gpu_num, (
            f"Batch size does not match GPU number: {len(self.data)} != {self.gpu_num}"
        )

        for batch in self.data:
            batch.verify()

    def _find_max_load_batch(self) -> Tuple[int, BatchedGRPOData]:
        """Find the index of the batch with maximum load that can provide sequences."""
        valid_batches = [
            (idx, batch) for idx, batch in enumerate(self.data) if batch.seq_num > 1
        ]
        return max(valid_batches, key=lambda x: x[1].current_load, default=(None, None))

    def _find_min_load_batch(self) -> Tuple[int, BatchedGRPOData]:
        """Find the index of the batch with minimum load."""
        return min(enumerate(self.data), key=lambda x: x[1].current_load)

    def _will_improve_balance(
        self, source_batch: BatchedGRPOData, target_batch: BatchedGRPOData
    ) -> bool:
        assert source_batch.seq_num > 1
        item_to_move = source_batch.data[-1]

        current_max_load = self.max_load

        source_new_longest = source_batch.data[0].length
        source_new_load = source_new_longest * (source_batch.seq_num - 1)

        target_new_longest = max(target_batch.longest_seq_len, item_to_move.length)
        target_new_load = target_new_longest * (target_batch.seq_num + 1)

        others_max_load = max(
            [
                b.current_load
                for b in self.data
                if b != source_batch and b != target_batch
            ],
            default=0,
        )

        new_max_load = max(source_new_load, target_new_load, others_max_load)
        return new_max_load < current_max_load

    def _execute_move(
        self, source_batch: BatchedGRPOData, target_batch: BatchedGRPOData
    ):
        item = source_batch.data.pop()
        target_batch.fit_in(item)
        return item

    def balance(self):
        assert self.total_seq_num >= self.gpu_num

        initial_max_load = self.max_load
        while True:
            source_idx, source_batch = self._find_max_load_batch()
            target_idx, target_batch = self._find_min_load_batch()

            if source_idx is None or source_idx == target_idx:
                break

            if target_batch.seq_num == 0:
                item = self._execute_move(source_batch, target_batch)
                logger.debug(
                    f"Move item: {item.prompt_idx}, {item.response_idx}, from batch {source_idx} to {target_idx}, Reason: Target batch is empty"
                )
            elif self._will_improve_balance(source_batch, target_batch):
                eff_before = self.efficiency
                item = self._execute_move(source_batch, target_batch)
                eff_after = self.efficiency
                assert eff_after >= eff_before
                logger.debug(
                    f"Move item: {item.prompt_idx}, {item.response_idx}, from batch {source_idx} to {target_idx}, Reason: Will improve balance, Efficiency before: {eff_before}, after: {eff_after}"
                )
            else:
                break

        assert self.max_load <= initial_max_load

    def log(self, prefix: str = "", idx: int = 0):
        logger.info(
            f"{prefix}MicroStep {idx} (gpu_num={self.gpu_num}, seq_num={self.total_seq_num}, efficiency={self.efficiency})"
        )
        for idx, item in enumerate(self.data):
            item.log(prefix + "    ", idx)


class GlobalStepGRPOData(BaseModel):
    token_budget: int = Field(..., description="Token budget per batch")
    gpu_num: int = Field(..., description="Number of GPUs")
    max_micro_step_num: int = Field(
        ..., gt=0, description="Maximum number of micro steps"
    )
    data: List[MicroStepGRPOData] = Field(
        default_factory=list, description="List of micro step data"
    )

    @property
    def is_empty(self) -> bool:
        return len(self.data) == 0

    @property
    def micro_step_num(self):
        return len(self.data)

    @property
    def total_seq_num(self) -> int:
        """The total number of sequences in the batch."""
        return sum(micro_step.total_seq_num for micro_step in self.data)

    @property
    def total_batch_num(self) -> int:
        """The total number of batches in the batch."""
        return self.micro_step_num * self.gpu_num

    @property
    def total_effective_load(self) -> int:
        """The total number of effective tokens in the batch."""
        return sum(micro_step.total_effective_load for micro_step in self.data)

    @property
    def max_budget(self) -> int:
        max_budget = 0
        for micro_step in self.data:
            max_budget = max(max_budget, micro_step.max_load)
        return max_budget

    @property
    def efficiency(self) -> float:
        assert len(self.data) != 0, "No micro steps available"
        micro_step_efficiency = [micro_step.efficiency for micro_step in self.data]
        return sum(micro_step_efficiency) / len(micro_step_efficiency)

    @property
    def extra_seq_num_from_previous_micro_steps(self) -> int:
        return sum(micro_step.total_extra_seq_num for micro_step in self.data[:-1])

    @property
    def seq_needed_num_from_last_micro_step(self) -> int:
        assert len(self.data) != 0, "No micro steps available"
        return self.data[-1].seq_needed_num

    def create_micro_step(self) -> bool:
        if self.micro_step_num >= self.max_micro_step_num:
            return False
        self.data.append(
            MicroStepGRPOData(gpu_num=self.gpu_num, token_budget=self.token_budget)
        )
        return True

    def fit_in(self, data: GRPOData) -> bool:
        assert data.length <= self.token_budget

        for micro_step in self.data:
            if micro_step.fit_in(data):
                return True
        if self.create_micro_step():
            return self.fit_in(data)
        return False

    def _pop_from_previous_micro_steps(self) -> GRPOData:
        for micro_step in reversed(self.data[:-1]):
            if micro_step.total_extra_seq_num > 0:
                return micro_step.pop()
        assert False, "Should not reach here if assertions above are correct"

    def can_be_balanced(self) -> bool:
        return (
            self.extra_seq_num_from_previous_micro_steps
            >= self.seq_needed_num_from_last_micro_step
        )

    def balance(self):
        assert self.can_be_balanced()

        while self.seq_needed_num_from_last_micro_step > 0:
            popped_data = self._pop_from_previous_micro_steps()
            self.data[-1].fit_in(popped_data)

        for micro_step in self.data:
            micro_step.balance()

    def discard_last_micro_step(self) -> MicroStepGRPOData:
        assert len(self.data) > 0, "No micro steps to discard"
        return self.data.pop()

    def log(self, prefix: str = ""):
        logger.info(
            f"{prefix}GlobalStepGRPOData(micro_step_num={self.micro_step_num}, total_seq_num={self.total_seq_num}, token_budget={self.token_budget}, gpu_num={self.gpu_num})"
        )
        for idx, item in enumerate(self.data):
            item.log(prefix + "    ", idx)

    def verify(self):
        for micro_step in self.data:
            micro_step.verify()

        seen_pairs = set()
        for micro_step in self.data:
            for batch in micro_step.data:
                for item in batch.data:
                    pair = (item.prompt_idx, item.response_idx)
                    assert pair not in seen_pairs, (
                        f"Duplicate data found: prompt_idx={item.prompt_idx}, response_idx={item.response_idx}"
                    )
                    seen_pairs.add(pair)

        return True


def save_grpo_data_to_json(
    data: List[List[GRPOData]],
    output_path: str,
    encoding: str = "utf-8",
    indent: int = 4,
) -> None:
    """
    Save List[List[GRPOData]] to a JSON file format.
    Optimized to extract common prompt per batch and include statistics.

    Args:
        data: List of lists of GRPOData to save
        output_path: Path to output JSON file
        encoding: File encoding (default: utf-8)
        indent: JSON indentation for readability (default: 4)
    """

    def process_content(content):
        """Process message content, truncating image base64 data"""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            processed_content = []
            for item in content:
                if isinstance(item, dict):
                    processed_item = item.copy()
                    if item.get("type") == "image" and "image" in item:
                        image_data = item["image"]
                        # Truncate base64 image data to first 50 characters
                        if len(image_data) > 50:
                            processed_item["image"] = image_data[:50] + "...[TRUNCATED]"
                    processed_content.append(processed_item)
                else:
                    processed_content.append(item)
            return processed_content
        else:
            return content

    def extract_prompt_and_response(conversation):
        """Extract prompt (all messages except last) and response (last message)"""
        messages = conversation.messages
        if len(messages) == 0:
            return [], None

        # Process prompt messages (all except last)
        prompt_messages = []
        for msg in messages[:-1]:
            processed_msg = {"role": msg.role, "content": process_content(msg.content)}
            prompt_messages.append(processed_msg)

        # Process response message (last one)
        last_msg = messages[-1]
        response = {"role": last_msg.role, "content": process_content(last_msg.content)}

        return prompt_messages, response

    def calculate_batch_statistics(batch: List[GRPOData]):
        """Calculate statistics for a batch"""
        if not batch:
            return {}

        # Collect all numerical data
        reward_sums = [item.reward_sum for item in batch]
        advantages = [item.advantage for item in batch if item.advantage is not None]
        response_lengths = [item.response_length for item in batch]
        prompt_lengths = [item.prompt_length for item in batch]

        # Collect all reward types
        all_reward_types = set()
        for item in batch:
            if item.rewards:
                all_reward_types.update(item.rewards.keys())

        # Calculate statistics for each reward type
        reward_stats = {}
        for reward_type in all_reward_types:
            values = [
                item.rewards.get(reward_type, 0) for item in batch if item.rewards
            ]
            if values:
                reward_stats[reward_type] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                }

        stats = {
            "reward_sum": {
                "mean": statistics.mean(reward_sums),
                "std": statistics.stdev(reward_sums) if len(reward_sums) > 1 else 0,
                "min": min(reward_sums),
                "max": max(reward_sums),
            },
            "response_length": {
                "mean": statistics.mean(response_lengths),
                "std": statistics.stdev(response_lengths)
                if len(response_lengths) > 1
                else 0,
                "min": min(response_lengths),
                "max": max(response_lengths),
            },
            "prompt_length": {
                "mean": statistics.mean(prompt_lengths),
                "std": statistics.stdev(prompt_lengths)
                if len(prompt_lengths) > 1
                else 0,
                "min": min(prompt_lengths),
                "max": max(prompt_lengths),
            },
            "individual_rewards": reward_stats,
        }

        if advantages:
            stats["advantage"] = {
                "mean": statistics.mean(advantages),
                "std": statistics.stdev(advantages) if len(advantages) > 1 else 0,
                "min": min(advantages),
                "max": max(advantages),
            }

        return stats

    # Process all data
    processed_data = {
        "export_info": {
            "timestamp": datetime.now().isoformat(),
            "total_batches": len(data),
            "total_items": sum(len(batch) for batch in data),
            "note": "Image base64 data truncated to 50 characters, ref_logprobs excluded, common prompts extracted",
        },
        "batches": [],
    }

    for batch_idx, batch in enumerate(data):
        if not batch:
            continue

        # Extract common prompt from first item
        first_item = batch[0]
        common_prompt, _ = extract_prompt_and_response(first_item.conversation)

        # Calculate batch statistics
        batch_stats = calculate_batch_statistics(batch)

        batch_data = {
            "batch_index": batch_idx,
            "batch_size": len(batch),
            "common_prompt": common_prompt,
            "solution": first_item.solution,  # Solution should be same for all items in batch
            "statistics": batch_stats,
            "items": [],
        }

        for item_idx, item in enumerate(batch):
            # Extract only the response part and metadata
            _, response = extract_prompt_and_response(item.conversation)

            item_data = {
                "item_index": item_idx,
                "prompt_idx": item.prompt_idx,
                "response_idx": item.response_idx,
                "stop_reason": item.stop_reason,
                "length": item.length,
                "response_length": item.response_length,
                "prompt_length": item.prompt_length,
                "group_resp_token_sum": item.group_resp_token_sum,
                "group_seq_num": item.group_seq_num,
                "response": response,
                "rewards": item.rewards,
                "reward_sum": item.reward_sum,
            }

            # Add optional fields if they exist
            if item.global_resp_token_sum is not None:
                item_data["global_resp_token_sum"] = item.global_resp_token_sum
            if item.global_seq_num is not None:
                item_data["global_seq_num"] = item.global_seq_num
            if item.global_group_num is not None:
                item_data["global_group_num"] = item.global_group_num
            if item.advantage is not None:
                item_data["advantage"] = item.advantage

            batch_data["items"].append(item_data)

        processed_data["batches"].append(batch_data)

    # Save to JSON file
    with open(output_path, "w", encoding=encoding) as f:
        json.dump(processed_data, f, indent=indent, ensure_ascii=False)

    logger.info(
        f"Successfully saved {len(data)} batches with {sum(len(batch) for batch in data)} total items to {output_path}"
    )
