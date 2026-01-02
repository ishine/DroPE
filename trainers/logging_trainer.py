import torch
import numpy as np
from transformers import Trainer
import inspect
from typing import Optional, List, Optional, Callable
from trl import SFTTrainer, GRPOTrainer


def is_tensor(t):
    if isinstance(t, torch.Tensor):
        return True
    return False


class TrainerWithModelMetrics(Trainer):
    def __init__(
        self,
        *args,
        init_callback_functions: Optional[
            List[str | Callable[..., None]]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.initialize_callbacks(init_callback_functions)

    def initialize_callbacks(
        self,
        init_callback_functions: Optional[List[str | Callable[..., None]]],
    ) -> None:
        if not init_callback_functions:
            self.logging_model = False
            return
        self.logging_true = False
        for fn in init_callback_functions:
            if isinstance(fn, str):
                if not hasattr(self.model, fn):
                    raise AttributeError(f"model has no attribute '{fn}'")
                fn = getattr(self.model, fn)
            if not callable(fn):
                raise TypeError(f"callback {fn!r} is not callable")

            if "trainer" not in inspect.signature(fn).parameters:
                raise TypeError(
                    f"callback {fn.__name__} must accept a 'trainer' keyword"
                )

            fn(trainer=self)

    def get_model_custom_metric(self,):
        model_dict = self.model.get_and_flush_metrics()
        logged_dict = {}
        for log_name, log_value in model_dict.items():
            if is_tensor(log_value):
                log_value = log_value.mean().item()
            elif isinstance(log_value, (list, tuple)):
                log_value = np.mean(log_value)
            else:
                log_value = float(log_value)
            logged_dict[log_name] = log_value
        return logged_dict

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        if self.logging_model:
            logs.update(self.get_model_custom_metric())
        super().log(logs, start_time=start_time)


class SFTTrainerWithModelMetrics(TrainerWithModelMetrics, SFTTrainer):
    pass


class GRPOTrainerWithModelMetrics(TrainerWithModelMetrics, GRPOTrainer):
    pass
