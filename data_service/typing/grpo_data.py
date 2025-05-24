from typing import List, Type, Optional, Union
import logging
from dataclasses import dataclass, field

from data_service.typing.messages import Conversation

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class GRPOData:
    prompt_idx: Union[str, int]
    response_idx: int

    conversation: Conversation
    solution: str
    stop_reason: str

    length: int
    ref_logprobs: List[float]

    group_resp_token_num: int
    group_seq_num: int
    rewards: Optional[List[float]] = None
    advantage: Optional[float] = None

    @property
    def response_length(self):
        return len(self.ref_logprobs)

    @property
    def prompt_length(self):
        return self.length - self.response_length

    def log(self, prefix: str = "", idx: int = 0):
        logger.info(
            f"{prefix}Data {idx} (prompt_idx={self.prompt_idx}, response_idx={self.response_idx}, length={self.length}/{self.prompt_length}/{self.response_length}, stop_reason={self.stop_reason})"
        )


@dataclass
class BatchedGRPOData:
    """A batch of GRPOData objects."""

    token_budget: int
    data: List[GRPOData] = field(default_factory=list)

    _item_type: Type = field(default=GRPOData, init=False, repr=False)

    def __post_init__(self):
        assert isinstance(self.data, list), "data must be a list"
        assert all(isinstance(item, GRPOData) for item in self.data), (
            "all items must be GRPOData instances"
        )

    @property
    def longest_seq_len(self) -> int:
        """The length of the longest sequence in the batch."""
        if not self.data:
            return 0
        return max(item.length for item in self.data)

    @property
    def seq_num(self) -> int:
        """The number of sequences in the batch."""
        return len(self.data)

    @property
    def extra_seq_num(self) -> int:
        """The number of extra sequences in the batch."""
        if self.seq_num == 0:
            return 0
        return self.seq_num - 1

    @property
    def current_load(self) -> int:
        """The current load of the batch."""
        return self.longest_seq_len * self.seq_num

    @property
    def effective_load(self) -> int:
        """The effective load of the batch."""
        if not self.data:
            return 0
        return sum(item.length for item in self.data)

    @property
    def efficiency(self) -> float:
        """The efficiency of the batch."""
        if not self.data:
            return 0
        return self.effective_load / self.token_budget

    def fit_in(self, data: GRPOData, add_to_batch: bool = True):
        """Fit an item into the batch."""
        if (
            max(self.longest_seq_len, data.length) * (self.seq_num + 1)
            > self.token_budget
        ):
            return False
        if add_to_batch:
            self.data.append(data)
        return True

    def verify(self):
        """Verify the batch."""
        if self.seq_num == 0:
            raise ValueError("Batch is empty")
        if self.current_load > self.token_budget:
            raise ValueError(
                f"Batch is over budget: {self.current_load} > {self.token_budget}"
            )

    def log(self, prefix: str = "", idx: int = 0):
        logger.info(
            f"{prefix}Batch {idx} (seq_num={self.seq_num}, token_budget={self.token_budget}, current_load={self.current_load}, efficiency={self.efficiency})"
        )
        for idx, item in enumerate(self.data):
            item.log(prefix + "    ", idx)


@dataclass
class MicroStepGRPOData:
    """A batch of GRPOData objects for a micro step."""

    gpu_num: int
    token_budget: int

    data: List[BatchedGRPOData] = field(default_factory=list)
    _item_type: Type = field(default=BatchedGRPOData, init=False, repr=False)

    def __post_init__(self):
        assert isinstance(self.data, list), "data must be a list"
        assert all(isinstance(item, BatchedGRPOData) for item in self.data), (
            "all items must be BatchedGRPOData instances"
        )

        while len(self.data) < self.gpu_num:
            self.data.append(BatchedGRPOData(token_budget=self.token_budget))

    def fit_in(self, data: GRPOData, add_to_batch: bool = True):
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
        return self.total_effective_load / (self.gpu_num * self.max_load)

    def pop(self, num: int = 1) -> List[GRPOData]:
        result = []
        remaining = num

        if num > self.total_extra_seq_num:
            raise ValueError(
                f"Not enough extra sequences: {num} > {self.total_extra_seq_num}"
            )

        for batch_idx in range(len(self.data) - 1, -1, -1):
            batch = self.data[batch_idx]

            while batch.seq_num > 0 and remaining > 0:
                item = batch.data.pop()
                result.append(item)
                remaining -= 1

                if remaining == 0:
                    break

            if remaining == 0:
                break

        return result

    def verify(self):
        if len(self.data) != self.gpu_num:
            raise ValueError(
                f"Batch size does not match GPU number: {len(self.data)} != {self.gpu_num}"
            )
        if self.total_seq_num < self.gpu_num:
            raise ValueError(
                f"Total sequence number ({self.total_seq_num}) is smaller than GPU count ({self.gpu_num})"
            )

        for batch in self.data:
            batch.verify()

    def _find_max_load_batch_idx(self):
        """Find the index of the batch with maximum load that can provide sequences."""
        max_load = -1
        max_load_idx = -1

        for idx, batch in enumerate(self.data):
            # Only consider batches with more than 1 sequence (can spare sequences)
            if batch.seq_num > 1 and batch.current_load > max_load:
                max_load = batch.current_load
                max_load_idx = idx

        return max_load_idx if max_load_idx >= 0 else None

    def _find_min_load_batch_idx(self):
        """Find the index of the batch with minimum load."""
        min_load = float("inf")
        min_load_idx = -1

        for idx, batch in enumerate(self.data):
            if batch.current_load < min_load:
                min_load = batch.current_load
                min_load_idx = idx

        return min_load_idx

    def _will_improve_balance(self, source_batch, target_batch, item):
        """
        Check if moving an item from source_batch to target_batch will improve balance.

        Args:
            source_batch: The batch to move from
            target_batch: The batch to move to
            item: The item to move

        Returns:
            bool: True if move improves balance, False otherwise
        """
        # Calculate current max load
        current_max_load = self.max_load

        # Calculate what would be the new loads after moving
        source_new_longest = max(
            [d.length for d in source_batch.data if d != item], default=0
        )
        source_new_load = source_new_longest * (source_batch.seq_num - 1)

        target_new_longest = max([d.length for d in target_batch.data] + [item.length])
        target_new_load = target_new_longest * (target_batch.seq_num + 1)

        # Calculate what would be the new max load
        all_loads = [b.current_load for b in self.data]
        all_loads.remove(source_batch.current_load)
        all_loads.remove(target_batch.current_load)
        all_loads.extend([source_new_load, target_new_load])
        new_max_load = max(all_loads)

        # Move improves balance if it reduces max load or adds to an empty batch
        return new_max_load < current_max_load or target_batch.seq_num == 0

    def balance(self):
        # First check that all conditions are met
        if self.total_seq_num < self.gpu_num:
            raise ValueError(
                f"Total sequence number ({self.total_seq_num}) is not greater than GPU count ({self.gpu_num})"
            )

        # Track initial max load for comparison
        initial_max_load = self.max_load
        improved = True

        while improved:
            improved = False

            # Find source (max load) and target (min load) batches
            source_idx = self._find_max_load_batch_idx()
            target_idx = self._find_min_load_batch_idx()

            # If no valid source found, we're done
            if source_idx is None:
                break

            # If source and target are the same, we're done
            if source_idx == target_idx:
                break

            source_batch = self.data[source_idx]
            target_batch = self.data[target_idx]

            # Find best item to move from source to target
            best_item_idx = None

            for item_idx, item in enumerate(source_batch.data):
                # Check if item would fit in target batch
                if target_batch.fit_in(item, add_to_batch=False):
                    # Check if moving this item would improve balance
                    if self._will_improve_balance(source_batch, target_batch, item):
                        best_item_idx = item_idx
                        break

            # If we found a good item to move, do it
            if best_item_idx is not None:
                eff_before = self.efficiency
                item = source_batch.data.pop(best_item_idx)
                target_batch.data.append(item)
                logger.debug(
                    f"Move item: {item.prompt_idx}, {item.response_idx}, from {source_batch.seq_num} to {target_batch.seq_num}"
                )
                eff_after = self.efficiency
                logger.debug(f"Efficiency before: {eff_before}, after: {eff_after}")
                improved = True

        # Check if overall balance improved
        return self.max_load < initial_max_load

    def log(self, prefix: str = "", idx: int = 0):
        logger.info(
            f"{prefix}MicroStep {idx} (gpu_num={self.gpu_num}, seq_num={self.total_seq_num}, efficiency={self.efficiency})"
        )
        for idx, item in enumerate(self.data):
            item.log(prefix + "    ", idx)


@dataclass
class GlobalStepGRPOData:
    """A collection of MicroStepGRPOData objects for a global step."""

    token_budget: int
    gpu_num: int
    max_micro_step_num: int

    data: List[MicroStepGRPOData] = field(default_factory=list)
    _item_type: Type = field(default=MicroStepGRPOData, init=False, repr=False)

    def __post_init__(self):
        assert isinstance(self.data, list), "data must be a list"
        assert all(isinstance(item, MicroStepGRPOData) for item in self.data), (
            "all items must be MicroStepGRPOData instances"
        )
        # FIXME: 这里使用断言似乎不太合适
        assert self.max_micro_step_num > 0, (
            "max_micro_step_num must be greater than 0"
        )

        if len(self.data) == 0:
            self.data.append(
                MicroStepGRPOData(gpu_num=self.gpu_num, token_budget=self.token_budget)
            )

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
    def ideal_average_effective_load_per_batch(self) -> int:
        """The ideal average effective load per batch."""
        return self.total_effective_load / self.total_batch_num

    @property
    def efficiency(self) -> float:
        micro_step_efficiency = [micro_step.efficiency for micro_step in self.data]
        return sum(micro_step_efficiency) / len(micro_step_efficiency)

    def add(self, data: GRPOData) -> Union[bool, Optional[str]]:
        if data.length > self.token_budget:
            return False, "Data length exceeds token budget"

        for micro_step in self.data[::-1]:
            if micro_step.fit_in(data):
                return True, None

        if self.micro_step_num < self.max_micro_step_num:
            self.data.append(
                MicroStepGRPOData(gpu_num=self.gpu_num, token_budget=self.token_budget)
            )
            self.data[-1].fit_in(data)
            return True, None
        else:
            return False, "Micro step number exceeds max micro step number"

    def balance(self):
        for micro_step in self.data:
            micro_step.balance()

    def log(self, prefix: str = ""):
        logger.info(
            f"{prefix}GlobalStepGRPOData(micro_step_num={self.micro_step_num}, total_seq_num={self.total_seq_num}, token_budget={self.token_budget}, gpu_num={self.gpu_num})"
        )
        for idx, item in enumerate(self.data):
            item.log(prefix + "    ", idx)

    def verify(self):
        """
        Verify that the global step data is valid.

        Checks:
        1. Total sequence number across all micro steps is greater than GPU count
        2. No duplicate data based on prompt_idx and response_idx

        Returns:
            bool: True if valid, False otherwise
        """
        # Check if total sequence number is greater than GPU count
        for micro_step in self.data:
            micro_step.verify()

        # Check for duplicates
        seen_pairs = set()
        for micro_step in self.data:
            for batch in micro_step.data:
                for item in batch.data:
                    pair = (item.prompt_idx, item.response_idx)
                    if pair in seen_pairs:
                        raise ValueError(
                            f"Duplicate data found: prompt_idx={item.prompt_idx}, response_idx={item.response_idx}"
                        )
                    seen_pairs.add(pair)

        return True
