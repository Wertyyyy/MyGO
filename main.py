from typing import List, Tuple
import logging
import os
import queue
import argparse
import importlib

import torch
import torch.optim as optim
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, gather_object
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    StateDictOptions,
)
import wandb
import swanlab
import yaml

from data_service.data_service import DataService
from utils import timer, Metrics, print_gathered_metrics

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def clip_listed_tensors(
    input_1: List[torch.Tensor], input_2: List[torch.Tensor]
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    result_1 = []
    result_2 = []

    for t1, t2 in zip(input_1, input_2):
        min_len = min(len(t1), len(t2))
        if len(t1) != len(t2):
            logger.error(f"Lengths of tensors are not equal: {len(t1)} != {len(t2)}")
        result_1.append(t1[0:min_len])
        result_2.append(t2[0:min_len])

    return result_1, result_2


def average_listed_tensors(input: List[torch.Tensor]) -> torch.Tensor:
    input_sum = sum([torch.sum(t) for t in input])
    input_count = sum([len(t) for t in input])
    return input_sum / input_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
    main_config = config["main"]

    # Initialize accelerate
    torch.distributed.init_process_group(backend="cuda:nccl,cpu:gloo")
    accelerator = Accelerator()

    model_module = importlib.import_module(main_config["model_impl"]["path"])
    policy = model_module.MainTrainingImpl(**main_config["model_impl"]["params"])

    optimizer = optim.AdamW(policy.model.parameters(), lr=main_config["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=main_config["total_steps"] * accelerator.num_processes,
        eta_min=main_config["lr"] * 0.1,
    )

    policy.model, optimizer, scheduler = accelerator.prepare(
        policy.model, optimizer, scheduler
    )

    if accelerator.is_main_process:
        os.makedirs(main_config["save_dir"], exist_ok=True)

        swanlab.login(api_key="VloLlvL5VZQ62pkyvvgxo", save=True)
        swanlab.sync_wandb()

        wandb.login(key="706ec3477b40053429ec99b8a01b0ba95b6f1787")
        wandb.init(
            project="qwen2-vl-grpo",
            name=main_config["run_name"],
        )

        data_service = DataService(**config["data_service"])
        data_fetch_queue = queue.Queue(maxsize=main_config["queue_depth"] + 1)

    accelerator.wait_for_everyone()

    for global_step in range(main_config["total_steps"]):
        if accelerator.is_main_process:
            logger.info(f"Global step: {global_step}")

        optimizer.zero_grad()
        metrics = Metrics()

        with timer("global_step", accelerator.process_index, metrics):
            accelerator.wait_for_everyone()
            with timer("gather_weights", accelerator.process_index, metrics):
                state_dict = get_model_state_dict(
                    model=policy.model,
                    options=StateDictOptions(full_state_dict=True),
                )

            accelerator.wait_for_everyone()
            if (
                accelerator.is_main_process
                and (global_step + 1) % main_config["save_steps"] == 0
            ):
                save_path = os.path.join(main_config["save_dir"], f"step_{global_step}")
                logger.info(f"Saving model {save_path} at step {global_step}")
                unwrapped_model = accelerator.unwrap_model(policy.model)
                unwrapped_model.save_pretrained(
                    save_path,
                    safe_serialization=True,
                    is_main_process=accelerator.is_main_process,
                    state_dict=state_dict,
                )
                policy.processor.save_pretrained(save_path)

            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                with timer("update_weights", accelerator.process_index, metrics):
                    data_service.update_vllm_weights(state_dict)

            accelerator.wait_for_everyone()
            del state_dict

            if accelerator.is_main_process:
                with timer("waiting_for_data", accelerator.process_index, metrics):
                    while not data_fetch_queue.full():
                        logging.info(
                            f"Adding new fetch task to the queue, current step {global_step}"
                        )
                        data_fetch_queue.put_nowait(data_service.fetch_data())
                    current_fetch_task = data_fetch_queue.get()
                    global_step_grpo_data = current_fetch_task.result(metrics)
            else:
                global_step_grpo_data = None

            accelerator.wait_for_everyone()
            with timer("broadcast_data", accelerator.process_index, metrics):
                global_step_grpo_data = broadcast_object_list(
                    [global_step_grpo_data], from_process=0
                )[0]

            accelerator.wait_for_everyone()
            for micro_batch_data in global_step_grpo_data.data:
                with timer("micro_batch", accelerator.process_index, metrics):
                    batch_data_per_gpu = micro_batch_data.data[
                        accelerator.process_index
                    ]

                    with timer("forward_pass", accelerator.process_index, metrics):
                        batched_policy_logprobs = policy.get_logprobs(
                            [data.conversation for data in batch_data_per_gpu.data],
                            max_length=main_config["max_length"],
                        )

                    with timer("loss_computation", accelerator.process_index, metrics):
                        batched_ref_logprobs = [
                            torch.tensor(data.ref_logprobs, device=accelerator.device)
                            for data in batch_data_per_gpu.data
                        ]
                        batched_policy_logprobs, batched_ref_logprobs = (
                            clip_listed_tensors(
                                batched_policy_logprobs, batched_ref_logprobs
                            )
                        )

                        batched_per_token_kl = []
                        for pol_lp, ref_lp in zip(
                            batched_policy_logprobs, batched_ref_logprobs
                        ):
                            per_token_kl = (
                                torch.exp(ref_lp - pol_lp) - (ref_lp - pol_lp) - 1
                            )
                            batched_per_token_kl.append(per_token_kl)

                        batched_per_token_loss = []
                        for pol_lp, per_token_kl, adv in zip(
                            batched_policy_logprobs,
                            batched_per_token_kl,
                            [data.advantage for data in batch_data_per_gpu.data],
                        ):
                            # Compute policy gradient term with clipping
                            # Calculate ratio between policy and reference logprobs
                            coef_1 = torch.exp(pol_lp - pol_lp.detach())
                            # Apply clipping to the ratio
                            coef_2 = torch.clamp(
                                coef_1,
                                1 - main_config["epsilon_low"],
                                1 + main_config["epsilon_high"],
                            )
                            # Take the minimum (pessimistic bound)
                            per_token_scaled_adv = torch.min(coef_1 * adv, coef_2 * adv)

                            # Combine with KL penalty
                            per_token_loss = -(
                                per_token_scaled_adv
                                - main_config["grpo_beta"] * per_token_kl
                            )
                            batched_per_token_loss.append(per_token_loss)

                        loss = average_listed_tensors(batched_per_token_loss)
                        # FIXME: loss for grad accu is not consistent for metric
                        loss = (
                            loss
                            * batch_data_per_gpu.effective_load
                            / global_step_grpo_data.ideal_average_effective_load_per_batch
                        ) / global_step_grpo_data.micro_step_num
                        average_kl = average_listed_tensors(batched_per_token_kl)
                        average_kl = (
                            average_kl
                            * batch_data_per_gpu.effective_load
                            / global_step_grpo_data.ideal_average_effective_load_per_batch
                        ) / global_step_grpo_data.micro_step_num

                    with timer("backward_pass", accelerator.process_index, metrics):
                        accelerator.backward(loss)

                    metrics.add("Train/loss", loss.item())
                    metrics.add("Train/kl", average_kl.item())

            # NOTE: Temporarily disabled grad clip due to busy network
            # with timer("grad_clip", accelerator.process_index, metrics):
            #     grad_norm = torch.nn.utils.clip_grad_norm_(
            #         policy.model.parameters(),
            #         max_norm=main_config["max_grad_norm"],
            #     )
            #     metrics.add("Train/grad_norm", grad_norm.item())

            accelerator.wait_for_everyone()
            with timer("optimizer_step", accelerator.process_index, metrics):
                optimizer.step()
                scheduler.step()
                metrics.add("Train/lr", optimizer.param_groups[0]["lr"])

        metrics = gather_object([metrics])
        if accelerator.is_main_process:
            print_gathered_metrics(metrics, step=global_step)

    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
