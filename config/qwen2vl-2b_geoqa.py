from config.utils import get_path

_model_impl = "implement.model.qwen_vl"
_model_name = "Qwen2VL-2B-Instruct"
_run_name = "test_fsdp2"
_min_pixels = 4 * 28 * 28
_max_pixels = 512 * 28 * 28

model = {
    "impl_path": _model_impl,
    "init_params": {
        "pretrained_model_name_or_path": get_path("model", _model_name),
    },
}

processor = {
    "impl_path": _model_impl,
    "init_params": {
        "pretrained_model_name_or_path": get_path("model", _model_name),
        "min_pixels": _min_pixels,
        "max_pixels": _max_pixels,
    },
}

dataset = {
    "train_impl_path": "implement.dataset.geoqa",
    "train_dataset_path": get_path("data", "GeoQA"),
    "test_impl_path": "implement.dataset.geoqa",
    "test_dataset_path": get_path("data", "GeoQA-Test"),
    "system_prompt_path": None,
    "template_path": None,
}

reward = {
    "accuracy": "implement.reward.acc_simple",
    # "format": "implement.reward.format",
}

network = {
    "vllm_port": 41000,
    "nccl_port": 42000,
    "tf_port": 40001,
    "data_port": 43000,
    "vllm_host": "0.0.0.0",
    "tf_host": "10.244.33.39",
    "data_host": "0.0.0.0",
}

tf_server = {
    "device": "cuda:0",
    "token_budget": 20000,
}

vllm_server = {
    "device": "cuda:7",
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
    "temperature": 1.0,
    "pregenerate_steps": 1,
    "use_ref": "auto",
    "global_batch_size": 16,
    "per_prompt_generation_count": 8,
}

length = {
    "max_response_length": 500,
    "max_prompt_length": 500,
    "max_length": 1024,
}

training = {
    "lr": 1e-6,
    "total_steps": 1000,
    "grpo_beta": 0.00,
    "save_dir": get_path("model", _run_name),
    "save_steps": 100,
    "project_name": "MyGO",
    "run_name": _run_name,
    "max_grad_norm": 1.0,
    "token_budget": 9000,
    "max_micro_step_num": 1,
    "loss_type": "global",
}
