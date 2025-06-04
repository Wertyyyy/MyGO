from typing import Callable, List, Union, Optional
import logging
import time
import traceback

import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def track_time(name: str, mode: str = "sum"):
    def decorator(func: Callable) -> Callable:
        def wrapper(trainer, *args, **kwargs):
            start_time = time.time()
            result = func(trainer, *args, **kwargs)
            elapsed = time.time() - start_time
            trainer.metrics.add(f"Time/{name}", elapsed, mode=mode)
            return result

        return wrapper

    return decorator


def track_metrics(
    names: Optional[Union[str, List[str]]] = None, prefix: str = "", mode: str = "avg"
):
    def decorator(func: Callable) -> Callable:
        def wrapper(trainer, *args, **kwargs):
            result = func(trainer, *args, **kwargs)

            nonlocal names
            if isinstance(names, str):
                names = [names]

            if isinstance(result, dict):
                if names:
                    for key in names:
                        if key in result:
                            metric_key = f"{prefix}/{key}" if prefix else key
                            trainer.metrics.add(metric_key, result[key], mode=mode)
                else:
                    for key in result:
                        metric_key = f"{prefix}/{key}" if prefix else key
                        trainer.metrics.add(metric_key, result[key], mode=mode)
            elif isinstance(result, (tuple, list)):
                if names:
                    if len(names) != len(result):
                        raise ValueError(
                            f"Length of names ({len(names)}) does not match length of result ({len(result)})"
                        )
                    for key, value in zip(names, result):
                        metric_key = f"{prefix}/{key}" if prefix else key
                        trainer.metrics.add(metric_key, value, mode=mode)
                else:
                    func_name = func.__name__
                    for idx, value in enumerate(result):
                        metric_key = (
                            f"{prefix}/{func_name}_{idx}"
                            if prefix
                            else f"{func_name}_{idx}"
                        )
                        trainer.metrics.add(metric_key, value, mode=mode)
            elif isinstance(result, (int, float)) and len(names) == 1:
                metric_key = f"{prefix}/{names[0]}" if prefix else names[0]
                trainer.metrics.add(metric_key, result, mode=mode)
            elif isinstance(result, torch.Tensor) and len(names) == 1:
                metric_key = f"{prefix}/{names[0]}" if prefix else names[0]
                trainer.metrics.add(metric_key, result.item(), mode=mode)
            else:
                logger.warning(
                    f"Cannot track return values: expected dict or single value for "
                    f"{names}, got {type(result).__name__}"
                )
            return result

        return wrapper

    return decorator


def on_main_process(func):
    def wrapper(trainer, *args, **kwargs):
        if trainer.accelerator.is_main_process:
            return func(trainer, *args, **kwargs)
        else:
            return None

    return wrapper


def per_certain_step(key: str):
    def decorator(func: Callable) -> Callable:
        def wrapper(trainer, *args, **kwargs):
            if (trainer.global_step + 1) % trainer.config.training[key] == 0:
                return func(trainer, *args, **kwargs)
            else:
                return None

        return wrapper

    return decorator


def clear_and_log_metrics(func: Callable) -> Callable:
    def wrapper(trainer, *args, **kwargs):
        trainer.metrics.clear()
        func(trainer, *args, **kwargs)
        trainer.metrics.gather_and_log(step=trainer.global_step)

    return wrapper


def catch_exception(func: Callable) -> Callable:
    def wrapper(trainer, *args, **kwargs):
        try:
            return func(trainer, *args, **kwargs)
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(f"Error in {func.__name__}: {e}\nTraceback:\n{tb_str}")
            return None

    return wrapper
