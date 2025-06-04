from config.utils import get_path

_model_impl = "implement.model.qwen"
_model_name = "Qwen2.5-1.5B-Instruct"
_project_name = "MyGO"
_run_name = "no_kl_global"
_max_length = 1024


_tf_port = 40000
_vllm_port = 41000
_nccl_port = 42000
_data_port = 43000

model = {
    "impl_path": _model_impl,
    "params": {
        "model_name_or_path": get_path("model", _model_name),
    },
}

processor = {
    "impl_path": _model_impl,
    "params": {
        "model_name_or_path": get_path("model", _model_name),
    },
}

tf_server = {
    "host": "0.0.0.0",
    "port": _tf_port,
    "gpu_id": 0,
    "token_budget": 45000,
    "max_length": _max_length,
}

vllm_server = {
    "host": "0.0.0.0",
    "port": _vllm_port,
    "nccl_port": _nccl_port,
    "gpu_id": -1,
    "llm_params": {
        "model": get_path("model", _model_name),
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.85,
        "enable_prefix_caching": False,
        "max_model_len": 8192,
        "enforce_eager": False,
    },
}

data_server = {
    "host": "0.0.0.0",
    "port": _data_port,
    "dataset": {
        "impl_path": "implement.dataset.gsm8k",
        "params": {
            "dataset_path": get_path("data", "gsm8k_train.jsonl"),
            "system_prompt_path": None,
            "prompt_template_path": None,
        },
    },
    "reward": {
        "accuracy": "implement.reward.acc_simple",
        # "format": "implement.reward.format",
    },
    "network_config": {
        "vllm_host": "0.0.0.0",
        "vllm_port": _vllm_port,
        "tf_host": "10.244.175.235",
        "tf_port": _tf_port,
    },
    "generation_config": {
        "global_batch_size": 16,
        "per_prompt_generation_count": 8,
        "max_response_length": 500,
        "max_prompt_length": 500,
        "temperature": 1.0,
        "pregenerate_steps": 1,
    },
}

training = {
    "lr": 1e-6,
    "total_steps": 1000,
    "grpo_beta": 0.0,
    "save_dir": get_path("model", _run_name),
    "save_steps": 100,
    "project_name": _project_name,
    "run_name": _run_name,
    "max_length": _max_length,
    "max_grad_norm": 1.0,
    "token_budget": 9000,
    "max_micro_step_num": 2,
    "loss_type": "global",
}
