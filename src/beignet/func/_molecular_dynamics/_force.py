from typing import Callable

import torch.func
from torch import Tensor


def force(fn: Callable[..., Tensor]) -> Callable[..., Tensor]:
    def _fn(_x: Tensor, *args, **kwargs) -> Tensor:
        return -fn(_x, *args, **kwargs)

    return torch.func.grad(_fn)
