import torch
from torch import Tensor


def _pad_end(input: Tensor, pad_len: int, dim: int = 0, value=None) -> Tensor:
    pad = [0] * 2 * input.ndim
    pad[-(2 * dim + 1)] = pad_len
    return torch.nn.functional.pad(input, pad, mode="constant", value=value)


def pad_to_target_length(
    input: Tensor, target_length: int, dim: int = 0, value=None
) -> Tensor:
    pad_len = target_length - input.shape[dim]
    if pad_len < 0:
        raise ValueError(f"{pad_len=} < 0, ({input.shape=}, {dim=}, {target_length=})")

    return _pad_end(input, pad_len, dim, value=value)
