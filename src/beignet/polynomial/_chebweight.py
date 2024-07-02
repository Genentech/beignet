import torch
from torch import Tensor


def chebweight(
    input: Tensor,
) -> Tensor:
    return 1.0 / (torch.sqrt(1.0 + input) * torch.sqrt(1.0 - input))
