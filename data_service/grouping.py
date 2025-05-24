from typing import List, Optional, Tuple
import logging

from data_service.typing.grpo_data import (
    GRPOData,
    GlobalStepGRPOData,
    MicroStepGRPOData,
    BatchedGRPOData,
)
from utils import Metrics

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def adaptive_grouping(
    all_data: List[List[GRPOData]],
    gpu_num: int,
    token_budget: int,
    max_micro_step_num: int,
    metrics: Metrics,
) -> GlobalStepGRPOData:
    global_step_data = GlobalStepGRPOData(
        gpu_num=gpu_num,
        token_budget=token_budget,
        max_micro_step_num=max_micro_step_num,
    )

    flatten_data: List[GRPOData] = []
    for grouped_data in all_data:
        flatten_data.extend(grouped_data)
    flatten_data.sort(key=lambda x: x.length, reverse=True)

    discard_data: List[Tuple[GRPOData, str]] = []
    for data in flatten_data:
        success, err_msg = global_step_data.add(data)
        if not success:
            discard_data.append((data, err_msg))

    last_micro_step = global_step_data.data[-1]
    if last_micro_step.seq_needed_num > 0:
        logger.debug(
            f"Last micro step does not has enough data, need {last_micro_step.seq_needed_num} more sequences"
        )

        total_extra_seq_num = sum(
            other_micro_step.total_extra_seq_num
            for other_micro_step in global_step_data.data[:-1]
        )
        if total_extra_seq_num >= last_micro_step.seq_needed_num:
            logger.debug("Try to balance the last micro step")
            micro_step_idx = -2
            while last_micro_step.seq_needed_num > 0:
                current_pop_num = min(
                    global_step_data.data[micro_step_idx].total_extra_seq_num,
                    last_micro_step.seq_needed_num,
                )
                poped_data = global_step_data.data[micro_step_idx].pop(current_pop_num)
                for data in poped_data:
                    last_micro_step.fit_in(data)
        else:
            logger.debug("Discarding the last micro step")
            for data in last_micro_step.flatten():
                discard_data.append((data, "Last micro step could not be balanced"))
            global_step_data.data = global_step_data.data[:-1]

    for data, err_msg in discard_data:
        logger.info(f"Discarding data: {data}, {err_msg}")

    num_before = global_step_data.total_seq_num
    global_step_data.balance()
    num_after = global_step_data.total_seq_num
    assert num_before == num_after
    global_step_data.verify()

    metrics.add("Grouping/efficiency", global_step_data.efficiency)
    metrics.add("Grouping/seq_num", global_step_data.total_seq_num)
    metrics.add("Grouping/micro_step_num", global_step_data.micro_step_num)

    return global_step_data


def naive_grouping(
    data: List[GRPOData],
    gpu_num: int,
    token_budget: int,
    max_micro_step_num: int,
):
    global_batch_size = len(data)
    assert global_batch_size % gpu_num == 0
    micro_batch_size = global_batch_size // gpu_num

    global_step_data: List[MicroStepGRPOData] = []
    for micro_batch_idx in range(micro_batch_size):
        micro_batch_data: List[BatchedGRPOData] = []
        for gpu_idx in range(gpu_num):
            micro_batch_data.append(
                BatchedGRPOData(
                    token_budget=token_budget,
                    data=data[micro_batch_idx * gpu_num + gpu_idx].data,
                )
            )
        global_step_data.append(
            MicroStepGRPOData(
                data=micro_batch_data,
                token_budget=token_budget,
                gpu_num=gpu_num,
            )
        )

    return GlobalStepGRPOData(
        gpu_num=gpu_num,
        token_budget=token_budget,
        max_micro_step_num=max_micro_step_num,
        data=global_step_data,
    )


if __name__ == "__main__":
    import random

    def create_mock_grpo_data(
        prompt_idx: int, response_idx: int, length: Optional[int] = None
    ):
        if length is None:
            length = random.randint(5, 2048)

        return GRPOData(
            prompt_idx=prompt_idx,
            response_idx=response_idx,
            messages=[{"role": "system", "content": "Mock message"}],
            solution="Mock solution",
            length=length,
            response_length=1,
            stop_reason="mock",
            ref_logprobs=[0.1, 0.2, 0.3],
        )

    def create_mock_list_grouped_grpo_data(
        num_groups: int, num_data_per_group: int
    ) -> List[GRPOData]:
        grouped_data: List[GRPOData] = []
        for i in range(num_groups):
            group_data = []
            for j in range(num_data_per_group):
                group_data.append(create_mock_grpo_data(i, j))
            grouped_data.append(GRPOData(group_data))
        return grouped_data

    while True:
        num_groups = random.randint(1, 16)
        num_data_per_group = random.randint(1, 16)
        gpu_num = random.randint(1, 8)
        token_budget = random.randint(30, 4096)
        max_micro_step_num = random.randint(1, 16)

        grouped_data = create_mock_list_grouped_grpo_data(
            num_groups, num_data_per_group
        )
        global_step_data = adaptive_grouping(
            grouped_data,
            gpu_num=gpu_num,
            token_budget=token_budget,
            max_micro_step_num=max_micro_step_num,
        )
    # global_step_data.log()
