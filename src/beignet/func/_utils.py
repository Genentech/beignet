import torch
from torch import Tensor


def maybe_downcast(x):
    if isinstance(x, Tensor) and x.dtype is torch.float64:
        return x

    return x.to(dtype=torch.float32)
