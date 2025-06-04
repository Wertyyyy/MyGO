import threading
from typing import List, Dict, Any, Optional, Union
from collections import defaultdict

import torch
import logging
import wandb

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class LocalMetrics:
    """
    Local metrics collection for a single process.
    Thread-safe and supports hierarchical metrics.
    """

    def __init__(self):
        self._data = defaultdict(dict)
        self._counts = defaultdict(dict)
        self._lock = threading.Lock()

    def add(self, key: str, value: Union[float, int, torch.Tensor], mode: str = "avg"):
        """
        Add a metric value with hierarchical key support.

        Args:
            key: Hierarchical key using '/' as separator (e.g., 'Train/loss')
            value: Value to add (supports tensors, will be converted to float)
            mode: How to handle existing values - 'replace', 'sum', 'avg', 'max', 'min', 'last'
        """
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                value = value.item()
            else:
                value = value.detach().cpu().numpy().mean()

        value = float(value)

        with self._lock:
            keys = key.split("/")
            data_dict = self._data
            count_dict = self._counts

            # Navigate to the right level
            for k in keys[:-1]:
                if k not in data_dict:
                    data_dict[k] = {}
                    count_dict[k] = {}
                data_dict = data_dict[k]
                count_dict = count_dict[k]

            last_key = keys[-1]

            if last_key not in data_dict or mode == "replace":
                data_dict[last_key] = value
                count_dict[last_key] = 1
            elif mode == "sum":
                data_dict[last_key] += value
                count_dict[last_key] += 1
            elif mode == "avg":
                current_count = count_dict[last_key]
                data_dict[last_key] = (data_dict[last_key] * current_count + value) / (
                    current_count + 1
                )
                count_dict[last_key] += 1
            elif mode == "max":
                data_dict[last_key] = max(data_dict[last_key], value)
                count_dict[last_key] += 1
            elif mode == "min":
                data_dict[last_key] = min(data_dict[last_key], value)
                count_dict[last_key] += 1
            elif mode == "last":
                data_dict[last_key] = value
                count_dict[last_key] += 1

    def get(self, key: str) -> Optional[float]:
        """Get a metric value by key."""
        with self._lock:
            keys = key.split("/")
            data_dict = self._data

            for k in keys:
                if k not in data_dict:
                    return None
                data_dict = data_dict[k]

            return data_dict if isinstance(data_dict, (int, float)) else None

    def get_all(self) -> Dict[str, Any]:
        """Get all metrics as a nested dictionary."""
        with self._lock:
            return dict(self._data)

    def clear(self):
        """Clear all metrics."""
        with self._lock:
            self._data.clear()
            self._counts.clear()

    def to_flat_dict(self) -> Dict[str, float]:
        """Convert hierarchical metrics to flat dictionary."""

        def _flatten(data, prefix=""):
            flat = {}
            for key, value in data.items():
                full_key = f"{prefix}/{key}" if prefix else key
                if isinstance(value, dict):
                    flat.update(_flatten(value, full_key))
                else:
                    flat[full_key] = value
            return flat

        with self._lock:
            return _flatten(self._data)


class MetricsManager:
    """
    Global metrics manager that handles multi-process synchronization
    and integrates with accelerate and wandb.
    """

    def __init__(self, accelerator=None, auto_log=True):
        self.accelerator = accelerator
        self.auto_log = auto_log
        self.local_metrics = LocalMetrics()
        self._step_counter = 0

    @property
    def is_main_process(self):
        return self.accelerator is None or self.accelerator.is_main_process

    def add(self, key: str, value: Union[float, int, torch.Tensor], mode: str = "avg"):
        """Add a metric to local collection."""
        self.local_metrics.add(key, value, mode)

    def get(self, key: str) -> Optional[float]:
        """Get a local metric value."""
        return self.local_metrics.get(key)

    def clear(self):
        """Clear local metrics."""
        self.local_metrics.clear()

    def gather_and_log(self, step: Optional[int] = None, prefix: str = ""):
        """
        Gather metrics from all processes and log to wandb.
        Only executes on main process.
        """
        if self.accelerator is None:
            # Single process mode
            local_data = self.local_metrics.to_flat_dict()
            data_list = [local_data]
        else:
            # Multi-process mode
            try:
                from accelerate.utils import gather_object

                local_data = self.local_metrics.to_flat_dict()
                data_list = gather_object([local_data])
            except ImportError:
                logger.warning("accelerate not available, using local metrics only")
                local_data = self.local_metrics.to_flat_dict()
                data_list = [local_data]

        if self.is_main_process:
            aggregated = self._aggregate_metrics(data_list)
            self._log_metrics(aggregated, step, prefix)
            self._print_metrics(aggregated, step)

        return aggregated if self.is_main_process else {}

    def _aggregate_metrics(self, data_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics from multiple processes."""
        all_keys = set()

        # Collect all keys
        for data_dict in data_list:
            all_keys.update(data_dict.keys())

        aggregated = {}
        for key in all_keys:
            values = []
            for data_dict in data_list:
                if key in data_dict:
                    values.append(data_dict[key])

            if values:
                aggregated[key] = sum(values) / len(values)

        return aggregated

    def _log_metrics(self, metrics: Dict[str, float], step: Optional[int], prefix: str):
        """Log metrics to wandb."""
        if wandb.run is not None and self.auto_log:
            log_dict = {}
            for key, value in metrics.items():
                log_key = f"{prefix}/{key}" if prefix else key
                log_dict[log_key] = value

            if step is not None:
                wandb.log(log_dict, step=step)
            else:
                wandb.log(log_dict)

    def _print_metrics(self, metrics: Dict[str, float], step: Optional[int]):
        """Print metrics in a tree format."""
        if not metrics:
            return

        # Build tree structure
        tree = {}
        for key, value in metrics.items():
            parts = key.split("/")
            current = tree
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value

        # Print tree
        step_info = f" (Step {step})" if step is not None else ""
        logger.info(f"Metrics{step_info}:")
        self._print_tree(tree)

    def _print_tree(self, tree: Dict, prefix: str = "", is_last: bool = True):
        """Print tree structure recursively."""
        items = list(tree.items())
        for i, (key, value) in enumerate(items):
            is_last_item = i == len(items) - 1
            branch = "└── " if is_last_item else "├── "

            if isinstance(value, dict):
                logger.info(f"{prefix}{branch}{key}")
                next_prefix = prefix + ("    " if is_last_item else "│   ")
                self._print_tree(value, next_prefix, is_last_item)
            else:
                logger.info(f"{prefix}{branch}{key}: {value:.4g}")
