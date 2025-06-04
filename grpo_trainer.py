from typing import List, Tuple
import logging
import os
import sys
import importlib

import torch
import torch.optim as optim

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import broadcast_object_list
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    StateDictOptions,
)
import wandb
import swanlab

from vllm_service.vllm_client import VLLMClient
from data_service.grouping import adaptive_grouping
from data_service.typing.grpo_data import save_grpo_data_to_json
from data_service.data_client import DataClient
from utils.metrics import MetricsManager
from utils.decorators import (
    track_time,
    track_metrics,
    on_main_process,
    per_certain_step,
    clear_and_log_metrics,
    catch_exception,
)
from config.utils import load_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Trainer:
    def __init__(self, config_file: str):
        self.config = load_config(config_file)
        self.accelerator = Accelerator(
            kwargs_handlers=[InitProcessGroupKwargs(backend="cuda:nccl,cpu:gloo")]
        )
        self.metrics = MetricsManager(accelerator=self.accelerator)
        self._setup_model_and_optimizer()
        self._setup_clients()
        self.global_step = 0

    def _setup_model_and_optimizer(self):
        model_module = importlib.import_module(self.config.model["impl_path"])
        self.policy = model_module.TFModelImpl(**self.config.model["params"])

        processor_module = importlib.import_module(self.config.processor["impl_path"])
        self.processor = processor_module.TFProcessorImpl(
            **self.config.processor["params"]
        )

        self.optimizer = optim.AdamW(
            self.policy.model.parameters(), lr=self.config.training["lr"]
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training["total_steps"] * self.accelerator.num_processes,
            eta_min=self.config.training["lr"] * 0.1,
        )

        self.policy.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.policy.model, self.optimizer, self.scheduler
        )

    @on_main_process
    def _setup_clients(self):
        self.data_client = DataClient(
            host="localhost", port=self.config.data_server["port"]
        )
        self.data_client.initialize()
        self.data_client.reset()

        self.vllm_client = VLLMClient(
            host=self.config.vllm_server["host"],
            server_port=self.config.vllm_server["port"],
            nccl_port=self.config.vllm_server["nccl_port"],
        )
        self.vllm_client.initialize_sync()
        self.vllm_client.init_nccl_sync()

        os.makedirs(self.config.training["save_dir"], exist_ok=True)
        swanlab.sync_wandb()
        wandb.init(
            project=self.config.training["project_name"],
            name=self.config.training["run_name"],
        )

    @staticmethod
    def _clip_listed_tensors(
        input_1: List[torch.Tensor], input_2: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        result_1 = []
        result_2 = []

        for t1, t2 in zip(input_1, input_2):
            min_len = min(len(t1), len(t2))
            if len(t1) != len(t2):
                logger.error(
                    f"Lengths of tensors are not equal: {len(t1)} != {len(t2)}"
                )
            result_1.append(t1[0:min_len])
            result_2.append(t2[0:min_len])

        return result_1, result_2

    @staticmethod
    def _sum_with_denominator(tensor_list, denominator_list):
        return sum([torch.sum(t / d) for t, d in zip(tensor_list, denominator_list)])

    @track_time("gather_weights")
    def _gather_weights(self):
        return get_model_state_dict(
            model=self.policy.model,
            options=StateDictOptions(full_state_dict=True),
        )

    @on_main_process
    @per_certain_step("save_steps")
    @track_time("save_model")
    def _save_model(self, state_dict: dict):
        save_path = os.path.join(
            self.config.training["save_dir"], f"step_{self.global_step}"
        )
        logger.info(f"Saving model {save_path} at step {self.global_step}")

        unwrapped_model = self.accelerator.unwrap_model(self.policy.model)
        unwrapped_model.save_pretrained(
            save_path,
            safe_serialization=True,
            is_main_process=self.accelerator.is_main_process,
            state_dict=state_dict,
        )
        self.processor.processor.save_pretrained(save_path)

    @on_main_process
    @track_time("weight_sync")
    def _update_vllm_weights(self, state_dict: dict):
        self.vllm_client.update_weights_nccl_sync(state_dict)

    @on_main_process
    @catch_exception
    @track_time("waiting_for_data")
    def _fetch_data(self):
        self.data_client.update_step(self.global_step)
        all_data = self.data_client.fetch_data(self.global_step, self.metrics)
        save_grpo_data_to_json(
            all_data,
            os.path.join(
                self.config.training["save_dir"],
                f"data_step_{self.global_step}.json",
            ),
        )
        global_step_grpo_data = adaptive_grouping(
            all_data,
            gpu_num=self.accelerator.num_processes,
            token_budget=self.config.training["token_budget"],
            max_micro_step_num=self.config.training.get("max_micro_step_num", 16),
            metrics=self.metrics,
        )
        return global_step_grpo_data

    @track_time("forward_pass")
    def _forward_pass(self, batch_data_per_gpu):
        inputs = self.processor.prepare_inputs(
            batch_data_per_gpu.get_data_fields("conversation"),
            max_length=self.config.training["max_length"],
        )
        outputs = self.policy.forward(inputs)

        batched_policy_resp_logits, batched_policy_resp_input_ids = (
            self.processor.get_batched_resp_logits_and_input_ids(inputs, outputs)
        )
        batched_policy_logprobs = self.processor.get_batched_logprobs(
            batched_policy_resp_logits, batched_policy_resp_input_ids
        )
        batched_ref_logprobs = batch_data_per_gpu.get_data_fields("ref_logprobs")
        batched_policy_logprobs, batched_ref_logprobs = self._clip_listed_tensors(
            batched_policy_logprobs, batched_ref_logprobs
        )

        # Compute KL divergence
        batched_per_token_kl = []
        for pol_lp, ref_lp in zip(batched_policy_logprobs, batched_ref_logprobs):
            per_token_kl = torch.exp(ref_lp - pol_lp) - (ref_lp - pol_lp) - 1
            per_token_kl = torch.clamp(per_token_kl, min=-10, max=10)
            batched_per_token_kl.append(per_token_kl)

        # Compute entropy
        batched_per_token_entropy = self.processor.get_batched_entropy(
            batched_policy_resp_logits
        )

        # Set tensor fields
        batch_data_per_gpu.set_data_fields("pol_logprobs", batched_policy_logprobs)
        batch_data_per_gpu.set_data_fields(
            "per_token_entropy", batched_per_token_entropy
        )
        batch_data_per_gpu.set_data_fields("per_token_kl", batched_per_token_kl)
        return inputs, outputs

    @track_metrics(names="loss_for_training", prefix="Train", mode="sum")
    def _compute_loss(self, batch_data_per_gpu):
        # Compute policy loss
        batched_per_token_loss = []
        for pol_lp, per_token_kl, adv in zip(
            batch_data_per_gpu.get_data_fields("pol_logprobs"),
            batch_data_per_gpu.get_data_fields("per_token_kl"),
            batch_data_per_gpu.get_data_fields("advantage"),
        ):
            per_token_scaled_adv = torch.exp(pol_lp - pol_lp.detach()) * adv
            per_token_loss = -(
                per_token_scaled_adv - self.config.training["grpo_beta"] * per_token_kl
            )
            batched_per_token_loss.append(per_token_loss)

        batch_data_per_gpu.set_data_fields("per_token_loss", batched_per_token_loss)

        # Compute weights based on loss type
        loss_type = self.config.training["loss_type"]
        if loss_type == "local":
            denominator_per_seq = [
                (data.response_length * data.global_seq_num)
                for data in batch_data_per_gpu.data
            ]
        elif loss_type == "group":
            denominator_per_seq = [
                (data.group_resp_token_sum * data.global_group_num)
                for data in batch_data_per_gpu.data
            ]
        elif loss_type == "global":
            denominator_per_seq = [
                (data.global_resp_max_len * data.global_seq_num)
                for data in batch_data_per_gpu.data
            ]
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")

        loss = self._sum_with_denominator(batched_per_token_loss, denominator_per_seq)
        return loss * self.accelerator.num_processes

    @track_metrics(names=["loss", "kl", "entropy"], prefix="Train", mode="sum")
    def _compute_statistics(self, batch_data_per_gpu):
        denominator_per_seq = [
            (data.response_length * data.global_seq_num)
            for data in batch_data_per_gpu.data
        ]
        loss_for_metric = self._sum_with_denominator(
            batch_data_per_gpu.get_data_fields("per_token_loss"), denominator_per_seq
        ) * self.accelerator.num_processes
        kl_for_metric = self._sum_with_denominator(
            batch_data_per_gpu.get_data_fields("per_token_kl"), denominator_per_seq
        ) * self.accelerator.num_processes
        entropy_for_metric = self._sum_with_denominator(
            batch_data_per_gpu.get_data_fields("per_token_entropy"), denominator_per_seq
        ) * self.accelerator.num_processes
        return loss_for_metric, kl_for_metric, entropy_for_metric

    @track_time("backward_pass")
    def _backward_pass(self, loss):
        self.accelerator.backward(loss)

    @track_time("grad_clip")
    @track_metrics(names="grad_norm", prefix="Train")
    def _grad_clip(self):
        return torch.nn.utils.clip_grad_norm_(
            self.policy.model.parameters(),
            max_norm=self.config.training["max_grad_norm"],
        )

    @track_time("optimizer_step")
    @track_metrics(names="lr", prefix="Train")
    def _optimizer_step(self):
        self.optimizer.step()
        self.scheduler.step()
        return self.scheduler.get_last_lr()[0]

    @track_time("micro_batch")
    def _process_micro_batch(self, batch_data_per_gpu):
        self._forward_pass(batch_data_per_gpu)
        loss = self._compute_loss(batch_data_per_gpu)
        self._backward_pass(loss)
        self._compute_statistics(batch_data_per_gpu)

    @clear_and_log_metrics
    @track_time("global_step")
    def train_step(self):
        self.optimizer.zero_grad()

        state_dict = self._gather_weights()
        self._save_model(state_dict)
        self._update_vllm_weights(state_dict)
        del state_dict

        global_step_grpo_data = self._fetch_data()
        global_step_grpo_data = broadcast_object_list(
            [global_step_grpo_data], from_process=0
        )[0]
        if global_step_grpo_data is None or global_step_grpo_data.is_empty:
            logger.info(
                f"No data for global step {self.global_step} which type is {type(global_step_grpo_data)}"
            )
            return

        for micro_batch_data in global_step_grpo_data.data:
            batch_data_per_gpu = micro_batch_data.data[self.accelerator.process_index]
            self._process_micro_batch(batch_data_per_gpu)

        self._grad_clip()
        self._optimizer_step()

    def train(self):
        self.accelerator.wait_for_everyone()

        for self.global_step in range(self.config.training["total_steps"]):
            self.train_step()

    @on_main_process
    def cleanup(self):
        self.vllm_client.close_sync()
        self.data_client.close()
        wandb.finish()
        swanlab.finish()


def main():
    trainer = Trainer(sys.argv[1])
    trainer.train()
    trainer.cleanup()


if __name__ == "__main__":
    main()
