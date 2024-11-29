import torch
from torch import Tensor


def maybe_downcast(x):
    if isinstance(x, Tensor) and x.dtype is torch.float64:
        return x

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.tensor(x).to(device=device, dtype=torch.float32)
