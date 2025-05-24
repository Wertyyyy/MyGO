import time
from contextlib import contextmanager
from typing import Optional, List, Dict
from collections import OrderedDict

import logging
import wandb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Metrics(OrderedDict):
    def add(self, key: str, value: float, mode: str = "avg"):
        """
        Add a metric value with hierarchical key support.

        Args:
            key: Hierarchical key using '/' as separator
            value: Value to add
            mode: How to handle existing values - 'replace', 'sum', or 'avg'
        """
        keys = key.split("/")
        d = self
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]

        last_key = keys[-1]
        if last_key in d and mode != "replace":
            if mode == "sum":
                d[last_key] += value
            elif mode == "avg":
                if not isinstance(d[last_key], dict):
                    if "_count" not in d:
                        d["_count"] = {}
                    if last_key not in d["_count"]:
                        d["_count"][last_key] = 1
                    d["_count"][last_key] += 1
                    d[last_key] = (
                        d[last_key] * (d["_count"][last_key] - 1) + value
                    ) / d["_count"][last_key]
            else:
                d[last_key] = value
        else:
            d[last_key] = value

    def add_dict(self, dict: Dict[str, float], mode: str = "avg"):
        for key, value in dict.items():
            self.add(key, value, mode=mode)


def print_gathered_metrics(gathered_metrics: List[Metrics], step: int):
    """
    Print metrics from all ranks in a tree-like format.
    First collects all metrics from all ranks, then calculates and prints the average values.
    Also logs metrics to wandb if step is provided.

    Args:
        gathered_metrics: List of Metrics objects from different ranks
        step: Current training step for wandb logging
    """
    if not gathered_metrics:
        logger.warning("No metrics to print")
        return

    # First, collect all keys from all ranks
    def collect_keys(data, prefix="", keys_set=None):
        if keys_set is None:
            keys_set = set()

        for key, value in data.items():
            if key == "_count":  # Skip count entries
                continue

            current_path = f"{prefix}/{key}" if prefix else key

            if isinstance(value, dict):
                collect_keys(value, current_path, keys_set)
            else:
                keys_set.add(current_path)

        return keys_set

    # Collect all unique keys
    all_keys = set()
    for metrics in gathered_metrics:
        all_keys.update(collect_keys(metrics))

    # Create a dictionary to store values from all ranks
    merged_data = {key: [] for key in all_keys}

    # Collect values for each key from all ranks
    def collect_values(data, prefix="", result_dict=None):
        for key, value in data.items():
            if key == "_count":  # Skip count entries
                continue

            current_path = f"{prefix}/{key}" if prefix else key

            if isinstance(value, dict):
                collect_values(value, current_path, result_dict)
            else:
                if current_path in result_dict:
                    result_dict[current_path].append(value)

    # Collect values from all ranks
    for metrics in gathered_metrics:
        collect_values(metrics, "", merged_data)

    # Function to build a tree structure from flat paths
    def build_tree(flat_dict):
        tree = {}
        for path, value in flat_dict.items():
            parts = path.split("/")
            current = tree
            for i, part in enumerate(parts):
                if i == len(parts) - 1:  # Last part
                    current[part] = value
                else:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
        return tree

    # Build a tree from the averages
    tree = build_tree(merged_data)

    # Log to wandb
    if wandb.run is not None:
        # Flatten the metrics for wandb logging
        wandb_metrics = {}

        def flatten_metrics(data, prefix=""):
            for key, value in data.items():
                if isinstance(value, dict):
                    flatten_metrics(value, f"{prefix}/{key}" if prefix else key)
                else:
                    wandb_metrics[f"{prefix}/{key}" if prefix else key] = float(
                        sum(value) / len(value)
                    )

        flatten_metrics(tree)
        wandb.log(wandb_metrics, step=step)

    # Print the tree recursively
    def print_tree(data, prefix="", is_last=True):
        for i, (key, value) in enumerate(data.items()):
            is_last_item = i == len(data) - 1
            branch = "└── " if is_last_item else "├── "

            if isinstance(value, dict):
                logger.info(f"{prefix}{branch}{key}")
                next_prefix = prefix + ("    " if is_last_item else "│   ")
                print_tree(value, next_prefix)
            else:
                avg_value = float(sum(value) / len(value))
                if len(value) > 1:
                    formatted_value = (
                        f"{avg_value:.4g}, {[f'{float(v):.4g}' for v in value]}"
                    )
                else:
                    formatted_value = f"{avg_value:.4g}"
                logger.info(f"{prefix}{branch}{key}: {formatted_value}")

    # Print the tree
    logger.info("Metrics averaged across all ranks:")
    print_tree(tree)


@contextmanager
def timer(name: str, rank: Optional[int] = 0, metrics: Optional[Metrics] = None):
    """
    Context manager to measure code execution time.
    Only prints the execution time when rank equals 0.

    Args:
        name: Name of the code block to measure
        rank: Current process rank (default: 0)
    """
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        if metrics is not None:
            metrics.add(f"Time/{name}", elapsed_time, mode="sum")
        if rank == 0:
            logger.debug(f"{name} took {elapsed_time:.4f} seconds")


if __name__ == "__main__":
    metrics_1 = Metrics()
    metrics_1.add("test/test", 1, mode="sum")
    metrics_1.add("test/test", 2, mode="sum")
    metrics_1.add("test/test", 3, mode="sum")
    metrics_1.add("test/aaa/bbb/ccc", 4, mode="sum")

    metrics_2 = Metrics()
    metrics_2.add("test/test", 1, mode="sum")
    metrics_2.add("test/test", 2, mode="sum")
    metrics_2.add("test/test", 3, mode="sum")
    metrics_2.add("test/aaa/bbb/ccc", 4, mode="sum")
    metrics_2.add("www", 5, mode="sum")

    metrics_3 = Metrics()
    metrics_3.add("test/test", 1, mode="sum")
    metrics_3.add("test/test", 2, mode="sum")
    metrics_3.add("test/test", 3, mode="sum")
    metrics_3.add("test/aaa/bbb/ccc", 4, mode="sum")
    metrics_3.add("test/aaa/bbb/ddd", 5, mode="sum")

    print_gathered_metrics([metrics_1, metrics_2, metrics_3], step=0)
