import logging
import os
import sys
import importlib

import torch
import torch.optim as optim

from accelerate import Accelerator, InitProcessGroupKwargs
from vllm_service.nccl_client import NCCLClient
from data_service.data_client import DataClient
from utils.metrics import MetricsManager
from utils.decorators import (
    track_time,
    track_metrics,
    track_memory,
    on_main_process,
    per_step,
    clear_and_log_metrics,
    catch_exception,
)
from config.utils import ConfigManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Trainer:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.accelerator = Accelerator(
            kwargs_handlers=[InitProcessGroupKwargs(backend="cuda:nccl,cpu:gloo")]
        )
        self.metrics = MetricsManager(accelerator=self.accelerator)
        self.data_client = DataClient(
            host="localhost", port=self.config.network.data_port
        )
        self.data_client.initialize()

        self._setup_clients()
        self._setup_model_and_optimizer()
        self.global_step = 0

    def _setup_model_and_optimizer(self):
        model_module = importlib.import_module(self.config.model.impl_path)
        init_params = self.config.model.init_params.to_dict()
        init_params["device_map"] = self.accelerator.device
        self.policy = model_module.TFModelImpl(init_params=init_params)

        processor_module = importlib.import_module(self.config.processor.impl_path)
        self.processor = processor_module.TFProcessorImpl(
            init_params=self.config.processor.init_params.to_dict()
        )

        self.optimizer = optim.AdamW(
            self.policy.model.parameters(), lr=self.config.training.lr
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training.total_steps * self.accelerator.num_processes,
            eta_min=self.config.training.lr * 0.1,
        )

        self.policy.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.policy.model, self.optimizer, self.scheduler
        )
        self.policy.model.set_reshard_after_backward(False)

    @on_main_process
    def _setup_clients(self):
        import wandb
        import swanlab

        cuda_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default")
        logger.info(f"PYTORCH_CUDA_ALLOC_CONF: {cuda_alloc_conf}")

        self.data_client.reset()

        self.nccl_client = NCCLClient(
            host=self.config.network.vllm_host,
            server_port=self.config.network.vllm_port,
            nccl_port=self.config.network.nccl_port,
            nccl_device=self.accelerator.device,
            dp_size=self.config.vllm_server.llm_params.data_parallel_size,
        )
        self.nccl_client.init_nccl()

        os.makedirs(self.config.training.save_dir, exist_ok=True)
        swanlab.sync_wandb()
        wandb.init(
            project=self.config.training.project_name,
            name=self.config.training.run_name,
        )

    @staticmethod
    def _sum_with_denominator(tensor_list, denominator_list):
        return sum([torch.sum(t / d) for t, d in zip(tensor_list, denominator_list)])

    def _get_state_dict_from_parameters(self):
        state_dict = {}
        for name, param in self.policy.model.named_parameters():
            name = name.replace("_checkpoint_wrapped_module.", "")
            state_dict[name] = param.data
            # logger.info(f"Parameter {name} with type {type(param.data)}")
            assert isinstance(param.data, torch.Tensor), (
                f"Parameter {name} with type {type(param.data)} is not a tensor"
            )

        return state_dict

    @on_main_process
    @per_step("training.save_steps")
    @track_time("save_model")
    def _save_model(self):
        save_path = os.path.join(
            self.config.training.save_dir, f"step_{self.global_step}"
        )
        logger.info(f"Saving model {save_path} at step {self.global_step}")

        state_dict = self._get_state_dict_from_parameters()

        self.policy.model.save_pretrained(save_path, state_dict=state_dict)
        self.processor.processor.save_pretrained(save_path)

    @on_main_process
    @track_time("weight_sync")
    def _update_vllm_weights(self):
        state_dict = self._get_state_dict_from_parameters()
        self.nccl_client.update_weights_nccl(state_dict)

    @catch_exception
    @track_time("waiting_for_data")
    def _fetch_data(self):
        rank_data = self.data_client.fetch_data(
            self.global_step,
            rank=self.accelerator.process_index,
            update_step=True,
        )
        reward = 0.0
        for data in rank_data:
            reward += sum(data.get_data_fields("reward_sum"))
        reward /= rank_data[0].data[0].global_seq_num
        self.metrics.add("Train/reward", reward, local_mode="sum", gather_mode="sum")
        return rank_data

    @track_time("forward_pass")
    def _forward_pass(self, batch_data_per_gpu):
        inputs = self.processor.prepare_inputs(
            batch_data_per_gpu.get_data_fields("conversation"),
            max_length=self.config.length.max_length,
        )
        outputs = self.policy.forward(inputs)

        # Compute policy logprobs
        batched_policy_logprobs = self.processor.get_batched_response_logprobs(
            inputs, outputs
        )
        batch_data_per_gpu.set_data_fields("pol_logprobs", batched_policy_logprobs)

        # Compute entropy
        batched_per_token_entropy = self.processor.get_batched_response_entropy(
            inputs, outputs
        )
        batch_data_per_gpu.set_data_fields(
            "per_token_entropy", batched_per_token_entropy
        )

        # Compute KL divergence
        batched_ref_logprobs = batch_data_per_gpu.get_data_fields("ref_logprobs")
        batched_per_token_kl = []
        for pol_lp, ref_lp in zip(batched_policy_logprobs, batched_ref_logprobs):
            if ref_lp is None:
                batched_per_token_kl.append(None)
                continue

            per_token_kl = torch.exp(ref_lp - pol_lp) - (ref_lp - pol_lp) - 1
            per_token_kl = torch.clamp(per_token_kl, min=-10, max=10)
            batched_per_token_kl.append(per_token_kl)

        batch_data_per_gpu.set_data_fields("per_token_kl", batched_per_token_kl)

        return inputs, outputs

    @track_metrics(names="loss_for_training", prefix="Train", local_mode="avg")
    def _compute_loss(self, batch_data_per_gpu):
        # Compute policy loss
        batched_per_token_loss = []
        for pol_lp, per_token_kl, adv in zip(
            batch_data_per_gpu.get_data_fields("pol_logprobs"),
            batch_data_per_gpu.get_data_fields("per_token_kl"),
            batch_data_per_gpu.get_data_fields("advantage"),
        ):
            per_token_scaled_adv = torch.exp(pol_lp - pol_lp.detach()) * adv
            if per_token_kl is not None:
                per_token_loss = -(
                    per_token_scaled_adv - self.config.training.grpo_beta * per_token_kl
                )
            else:
                per_token_loss = -per_token_scaled_adv
            batched_per_token_loss.append(per_token_loss)

        batch_data_per_gpu.set_data_fields("per_token_loss", batched_per_token_loss)

        # Compute weights based on loss type
        loss_type = self.config.training.loss_type
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

    @track_metrics(names=["loss", "kl", "entropy"], prefix="Train")
    def _compute_statistics(self, batch_data_per_gpu):
        denominator_per_seq = [
            (data.response_length * data.global_seq_num)
            for data in batch_data_per_gpu.data
        ]
        loss_for_metric = self._sum_with_denominator(
            batch_data_per_gpu.get_data_fields("per_token_loss"),
            denominator_per_seq,
        )
        batched_per_token_kl = batch_data_per_gpu.get_data_fields("per_token_kl")
        all_has_ref_lp = all(kl is not None for kl in batched_per_token_kl)
        if all_has_ref_lp:
            kl_for_metric = self._sum_with_denominator(
                batched_per_token_kl, denominator_per_seq
            )
        else:
            kl_for_metric = 0

        entropy_for_metric = self._sum_with_denominator(
            batch_data_per_gpu.get_data_fields("per_token_entropy"),
            denominator_per_seq,
        )
        return loss_for_metric, kl_for_metric, entropy_for_metric

    @track_time("backward_pass")
    @track_memory("backward_pass")
    def _backward_pass(self, loss):
        self.accelerator.backward(loss)

    @track_time("grad_clip")
    @track_metrics(names="grad_norm", prefix="Train")
    def _grad_clip(self):
        return torch.nn.utils.clip_grad_norm_(
            self.policy.model.parameters(),
            max_norm=self.config.training.max_grad_norm,
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

        rank_data = self._fetch_data()
        for batch in rank_data:
            self._process_micro_batch(batch)
        self._grad_clip()
        self._optimizer_step()

        self._update_vllm_weights()
        self._save_model()

    def train(self):
        self.accelerator.wait_for_everyone()

        for self.global_step in range(self.config.training.total_steps):
            self.train_step()

    @on_main_process
    def cleanup(self):
        import wandb

        self.nccl_client.close()
        self.data_client.close()
        wandb.finish()


def main():
    config = ConfigManager(sys.argv[1])
    trainer = Trainer(config)
    trainer.train()
    trainer.cleanup()


if __name__ == "__main__":
    main()
