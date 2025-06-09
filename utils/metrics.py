import threading
from typing import List, Dict, Any, Optional, Union
from collections import OrderedDict

import torch
import logging
import wandb

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class OrderedDefaultDict(OrderedDict):
    """OrderedDict with defaultdict-like behavior."""

    def __init__(self, default_factory=None):
        super().__init__()
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value


class LocalMetrics:
    """
    Local metrics collection for a single process.
    Thread-safe and supports hierarchical metrics with dual-mode control.
    Maintains insertion order for consistent display and gathering.
    """

    def __init__(self):
        self._data = OrderedDefaultDict(lambda: OrderedDict())
        self._counts = OrderedDefaultDict(lambda: OrderedDict())
        self._local_modes = OrderedDefaultDict(
            lambda: OrderedDict()
        )  # Store local aggregation modes
        self._gather_modes = OrderedDefaultDict(
            lambda: OrderedDict()
        )  # Store cross-rank gather modes
        self._lock = threading.Lock()

    def add(
        self,
        key: str,
        value: Union[float, int, torch.Tensor],
        local_mode: str = "avg",
        gather_mode: str = "avg",
    ):
        """
        Add a metric value with hierarchical key support and dual-mode control.

        Args:
            key: Hierarchical key using '/' as separator (e.g., 'Train/loss')
            value: Value to add (supports tensors, will be converted to float)
            local_mode: How to handle multiple adds locally - 'sum', 'avg', 'max', 'min'
            gather_mode: How to handle cross-rank aggregation - 'sum', 'avg', 'max', 'min'
        """
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                value = value.item()
            else:
                value = value.detach().cpu().numpy().mean()

        value = float(value)

        # Validate modes
        valid_modes = {"sum", "avg", "max", "min"}
        if local_mode not in valid_modes:
            raise ValueError(
                f"Invalid local_mode: {local_mode}. Must be one of {valid_modes}"
            )
        if gather_mode not in valid_modes:
            raise ValueError(
                f"Invalid gather_mode: {gather_mode}. Must be one of {valid_modes}"
            )

        with self._lock:
            keys = key.split("/")
            data_dict = self._data
            count_dict = self._counts
            local_mode_dict = self._local_modes
            gather_mode_dict = self._gather_modes

            # Navigate to the right level, creating OrderedDicts as needed
            for k in keys[:-1]:
                if k not in data_dict:
                    data_dict[k] = OrderedDict()
                    count_dict[k] = OrderedDict()
                    local_mode_dict[k] = OrderedDict()
                    gather_mode_dict[k] = OrderedDict()
                data_dict = data_dict[k]
                count_dict = count_dict[k]
                local_mode_dict = local_mode_dict[k]
                gather_mode_dict = gather_mode_dict[k]

            last_key = keys[-1]

            # Check for mode inconsistency and warn
            if last_key in local_mode_dict:
                if local_mode_dict[last_key] != local_mode:
                    logger.warning(
                        f"Local mode inconsistency for key '{key}': "
                        f"existing={local_mode_dict[last_key]}, new={local_mode}. "
                        f"Keeping existing mode."
                    )
                    local_mode = local_mode_dict[last_key]

                if gather_mode_dict[last_key] != gather_mode:
                    logger.warning(
                        f"Gather mode inconsistency for key '{key}': "
                        f"existing={gather_mode_dict[last_key]}, new={gather_mode}. "
                        f"Keeping existing mode."
                    )
                    gather_mode = gather_mode_dict[last_key]
            else:
                # First time setting modes for this key
                local_mode_dict[last_key] = local_mode
                gather_mode_dict[last_key] = gather_mode

            # Apply local aggregation based on local_mode
            if last_key not in data_dict:
                data_dict[last_key] = value
                count_dict[last_key] = 1
            else:
                if local_mode == "sum":
                    data_dict[last_key] += value
                    count_dict[last_key] += 1
                elif local_mode == "avg":
                    current_count = count_dict[last_key]
                    data_dict[last_key] = (
                        data_dict[last_key] * current_count + value
                    ) / (current_count + 1)
                    count_dict[last_key] += 1
                elif local_mode == "max":
                    data_dict[last_key] = max(data_dict[last_key], value)
                    count_dict[last_key] += 1
                elif local_mode == "min":
                    data_dict[last_key] = min(data_dict[last_key], value)
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

    def get_modes(self, key: str) -> Optional[Dict[str, str]]:
        """Get the local and gather modes for a key."""
        with self._lock:
            keys = key.split("/")
            local_mode_dict = self._local_modes
            gather_mode_dict = self._gather_modes

            for k in keys:
                if k not in local_mode_dict:
                    return None
                local_mode_dict = local_mode_dict[k]
                gather_mode_dict = gather_mode_dict[k]

            if isinstance(local_mode_dict, str) and isinstance(gather_mode_dict, str):
                return {"local_mode": local_mode_dict, "gather_mode": gather_mode_dict}
            return None

    def get_all(self) -> Dict[str, Any]:
        """Get all metrics as a nested ordered dictionary."""
        with self._lock:
            return OrderedDict(self._data)

    def get_all_modes(self) -> Dict[str, Dict[str, str]]:
        """Get all modes as nested ordered dictionaries."""
        with self._lock:
            return {
                "local_modes": OrderedDict(self._local_modes),
                "gather_modes": OrderedDict(self._gather_modes),
            }

    def clear(self):
        """Clear all metrics and modes."""
        with self._lock:
            self._data.clear()
            self._counts.clear()
            self._local_modes.clear()
            self._gather_modes.clear()

    def to_flat_dict(self) -> OrderedDict:
        """Convert hierarchical metrics to flat ordered dictionary."""

        def _flatten(data, prefix=""):
            flat = OrderedDict()
            for key, value in data.items():
                full_key = f"{prefix}/{key}" if prefix else key
                if isinstance(value, (dict, OrderedDict)):
                    flat.update(_flatten(value, full_key))
                else:
                    flat[full_key] = value
            return flat

        with self._lock:
            return _flatten(self._data)

    def to_flat_modes(self) -> OrderedDict:
        """Convert hierarchical modes to flat ordered dictionary."""

        def _flatten_modes(local_modes, gather_modes, prefix=""):
            flat = OrderedDict()
            for key in local_modes.keys():
                full_key = f"{prefix}/{key}" if prefix else key
                if isinstance(local_modes[key], (dict, OrderedDict)):
                    flat.update(
                        _flatten_modes(local_modes[key], gather_modes[key], full_key)
                    )
                else:
                    flat[full_key] = {
                        "local_mode": local_modes[key],
                        "gather_mode": gather_modes[key],
                    }
            return flat

        with self._lock:
            return _flatten_modes(self._local_modes, self._gather_modes)


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

    def add(
        self,
        key: str,
        value: Union[float, int, torch.Tensor],
        local_mode: str = "avg",
        gather_mode: str = "avg",
    ):
        """Add a metric to local collection with dual-mode control."""
        self.local_metrics.add(key, value, local_mode, gather_mode)

    def get(self, key: str) -> Optional[float]:
        """Get a local metric value."""
        return self.local_metrics.get(key)

    def get_modes(self, key: str) -> Optional[Dict[str, str]]:
        """Get the modes for a key."""
        return self.local_metrics.get_modes(key)

    def clear(self):
        """Clear local metrics."""
        self.local_metrics.clear()

    def gather_and_log(self, step: Optional[int] = None, prefix: str = ""):
        """
        Gather metrics from all processes using gather_mode and log to wandb.
        Only executes on main process.
        """
        if self.accelerator is None:
            # Single process mode
            local_data = self.local_metrics.to_flat_dict()
            local_modes = self.local_metrics.to_flat_modes()
            data_list = [local_data]
            modes_list = [local_modes]
        else:
            from accelerate.utils import gather_object

            local_data = self.local_metrics.to_flat_dict()
            local_modes = self.local_metrics.to_flat_modes()
            data_list = gather_object([local_data])
            modes_list = gather_object([local_modes])

        if self.is_main_process:
            aggregated = self._aggregate_metrics(data_list, modes_list)
            self._log_metrics(aggregated, step, prefix)
            self._print_metrics(aggregated, step)

        return aggregated if self.is_main_process else OrderedDict()

    def _aggregate_metrics(
        self, data_list: List[OrderedDict], modes_list: List[OrderedDict]
    ) -> OrderedDict:
        """Aggregate metrics from multiple processes using gather_mode while maintaining stable order."""
        # Collect all keys from all processes and maintain a stable order
        # Use the first process's key order as the canonical order, then add any missing keys alphabetically
        all_keys_ordered = []
        all_keys_set = set()

        # First, add keys in the order they appear in the first process
        if data_list:
            for key in data_list[0].keys():
                all_keys_ordered.append(key)
                all_keys_set.add(key)

        # Then add any additional keys from other processes in alphabetical order
        additional_keys = set()
        for data_dict in data_list[1:]:
            for key in data_dict.keys():
                if key not in all_keys_set:
                    additional_keys.add(key)

        # Add additional keys in sorted order for consistency
        for key in sorted(additional_keys):
            all_keys_ordered.append(key)
            all_keys_set.add(key)

        # Get gather modes from the first process (should be consistent across processes)
        gather_modes = modes_list[0] if modes_list else OrderedDict()

        aggregated = OrderedDict()
        for key in all_keys_ordered:
            values = []
            for data_dict in data_list:
                if key in data_dict:
                    values.append(data_dict[key])

            if values:
                # Get gather mode for this key, default to 'avg'
                gather_mode = gather_modes.get(key, {}).get("gather_mode", "avg")

                if gather_mode == "sum":
                    aggregated[key] = sum(values)
                elif gather_mode == "avg":
                    aggregated[key] = sum(values) / len(values)
                elif gather_mode == "max":
                    aggregated[key] = max(values)
                elif gather_mode == "min":
                    aggregated[key] = min(values)
                else:
                    logger.warning(
                        f"Unknown gather_mode '{gather_mode}' for key '{key}', using avg"
                    )
                    aggregated[key] = sum(values) / len(values)

        return aggregated

    def _log_metrics(self, metrics: OrderedDict, step: Optional[int], prefix: str):
        """Log metrics to wandb."""
        if wandb.run is not None and self.auto_log:
            log_dict = OrderedDict()
            for key, value in metrics.items():
                log_key = f"{prefix}/{key}" if prefix else key
                log_dict[log_key] = value

            if step is not None:
                wandb.log(log_dict, step=step)
            else:
                wandb.log(log_dict)

    def _print_metrics(self, metrics: OrderedDict, step: Optional[int]):
        """Print metrics in a tree format maintaining order."""
        if not metrics:
            return

        # Build tree structure while maintaining order
        tree = OrderedDict()
        for key, value in metrics.items():
            parts = key.split("/")
            current = tree
            for part in parts[:-1]:
                if part not in current:
                    current[part] = OrderedDict()
                current = current[part]
            current[parts[-1]] = value

        # Print tree
        step_info = f" (Step {step})" if step is not None else ""
        logger.info(f"Metrics{step_info}:")
        self._print_tree(tree)

    def _print_tree(self, tree: OrderedDict, prefix: str = "", is_last: bool = True):
        """Print tree structure recursively maintaining order."""
        items = list(tree.items())
        for i, (key, value) in enumerate(items):
            is_last_item = i == len(items) - 1
            branch = "└── " if is_last_item else "├── "

            if isinstance(value, (dict, OrderedDict)):
                logger.info(f"{prefix}{branch}{key}")
                next_prefix = prefix + ("    " if is_last_item else "│   ")
                self._print_tree(value, next_prefix, is_last_item)
            else:
                logger.info(f"{prefix}{branch}{key}: {value:.4g}")
