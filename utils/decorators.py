from typing import Callable, List, Union, Optional
import logging
import time
import traceback
from functools import partial, wraps
import gc

import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def track_time(name: str, local_mode: str = "sum", gather_mode: str = "avg"):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(trainer, *args, **kwargs):
            start_time = time.time()
            result = func(trainer, *args, **kwargs)
            elapsed = time.time() - start_time
            trainer.metrics.add(
                f"Time/{name}", elapsed, local_mode=local_mode, gather_mode=gather_mode
            )
            return result

        return wrapper

    return decorator


# FIXME: Support nested calls
def track_memory(name: str, local_mode: str = "max", gather_mode: str = "max"):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(trainer, *args, **kwargs):
            if not torch.cuda.is_available():
                return func(trainer, *args, **kwargs)

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            start_reserved = torch.cuda.memory_reserved() / (1024**3)

            result = func(trainer, *args, **kwargs)

            end_reserved = torch.cuda.memory_reserved() / (1024**3)
            peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)

            trainer.metrics.add(
                f"Memory/{name}/start",
                start_reserved,
                local_mode=local_mode,
                gather_mode=gather_mode,
            )
            trainer.metrics.add(
                f"Memory/{name}/end",
                end_reserved,
                local_mode=local_mode,
                gather_mode=gather_mode,
            )
            trainer.metrics.add(
                f"Memory/{name}/peak",
                peak_reserved,
                local_mode=local_mode,
                gather_mode=gather_mode,
            )

            return result

        return wrapper

    return decorator


def track_metrics(
    names: Optional[Union[str, List[str]]] = None,
    prefix: str = "",
    local_mode: str = "sum",
    gather_mode: str = "sum",
):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(trainer, *args, **kwargs):
            result = func(trainer, *args, **kwargs)

            nonlocal names
            if isinstance(names, str):
                names = [names]

            add_method = partial(
                trainer.metrics.add,
                local_mode=local_mode,
                gather_mode=gather_mode,
            )

            if isinstance(result, dict):
                if names:
                    for key in names:
                        if key in result:
                            metric_key = f"{prefix}/{key}" if prefix else key
                            add_method(metric_key, result[key])
                else:
                    for key in result:
                        metric_key = f"{prefix}/{key}" if prefix else key
                        add_method(metric_key, result[key])
            elif isinstance(result, (tuple, list)):
                if names:
                    if len(names) != len(result):
                        raise ValueError(
                            f"Length of names ({len(names)}) does not match length of result ({len(result)})"
                        )
                    for key, value in zip(names, result):
                        metric_key = f"{prefix}/{key}" if prefix else key
                        add_method(metric_key, value)
                else:
                    func_name = func.__name__
                    for idx, value in enumerate(result):
                        metric_key = (
                            f"{prefix}/{func_name}_{idx}"
                            if prefix
                            else f"{func_name}_{idx}"
                        )
                        add_method(metric_key, value)
            elif isinstance(result, (int, float)) and len(names) == 1:
                metric_key = f"{prefix}/{names[0]}" if prefix else names[0]
                add_method(metric_key, result)
            elif isinstance(result, torch.Tensor) and len(names) == 1:
                metric_key = f"{prefix}/{names[0]}" if prefix else names[0]
                add_method(metric_key, result.item())
            else:
                logger.warning(
                    f"Cannot track return values: expected dict or single value for "
                    f"{names}, got {type(result).__name__}"
                )
            return result

        return wrapper

    return decorator


def on_main_process(func):
    @wraps(func)
    def wrapper(trainer, *args, **kwargs):
        if trainer.accelerator.is_main_process:
            return func(trainer, *args, **kwargs)
        else:
            return None

    return wrapper


def per_step(key: str):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(trainer, *args, **kwargs):
            if (trainer.global_step + 1) % eval(f"trainer.config.{key}") == 0:
                return func(trainer, *args, **kwargs)
            else:
                return None

        return wrapper

    return decorator


def clear_and_log_metrics(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(trainer, *args, **kwargs):
        trainer.metrics.clear()
        func(trainer, *args, **kwargs)
        trainer.metrics.gather_and_log(step=trainer.global_step)

    return wrapper


def catch_exception(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(trainer, *args, **kwargs):
        try:
            return func(trainer, *args, **kwargs)
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(f"Error in {func.__name__}: {e}\nTraceback:\n{tb_str}")
            return None

    return wrapper
