import os
import importlib.util
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_path(type: str, path: str):
    if type == "model":
        return os.path.join(os.environ["MYGO_MODEL_PATH"], path)
    elif type == "data":
        return os.path.join(os.environ["MYGO_DATASET_PATH"], path)
    elif type == "prompt":
        return os.path.join(os.environ["MYGO_PROMPT_PATH"], path)
    else:
        raise ValueError(f"Invalid type: {type}")

def load_config(config_file: str):
    logger.info(f"Loading config from {config_file}")
    spec = importlib.util.spec_from_file_location("config_module", config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module