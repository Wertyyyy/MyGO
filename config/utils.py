import os
import importlib.util
import logging
import json
import yaml
from typing import Dict, Any
import copy

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


class ConfigManager:
    def __init__(self, config_file: str):
        logger.info(f"Loading config from {config_file}")
        spec = importlib.util.spec_from_file_location("config_module", config_file)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)

        self._config = {}
        for attr_name in dir(config_module):
            if not attr_name.startswith("_"):
                attr_value = getattr(config_module, attr_name)
                if isinstance(attr_value, dict):
                    self._config[attr_name] = attr_value

    def __getattr__(self, name: str):
        if name.startswith("_"):
            return super().__getattribute__(name)

        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                sub_config = ConfigManager.__new__(ConfigManager)
                sub_config._config = value
                return sub_config
            else:
                return value
        else:
            raise AttributeError(f"Config has no attribute '{name}'")

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._config)

    def save_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                self._config, f, indent=4, ensure_ascii=False, default=str
            )
        logger.info(f"Config saved to JSON file: {path}")

    def save_yaml(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                self._config,
                f,
                default_flow_style=False,
                allow_unicode=True,
                indent=4,
            )
        logger.info(f"Config saved to YAML file: {path}")