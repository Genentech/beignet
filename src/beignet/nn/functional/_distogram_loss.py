# ruff: noqa: E501

from typing import Literal

import torch
from torch import Tensor


def distogram_loss(input: Tensor, target: Tensor, mask: Tensor, start: float = 2.3125, end: float = 21.6875, steps: int = 64,     reduction: Literal["mean", "sum"] | None = "mean") -> Tensor:  # fmt: off
    target = torch.nn.functional.one_hot(torch.sum(torch.sum((target[..., None, :] - target[..., None, :, :]) ** 2.0, dim=-1, keepdim=True) > torch.linspace(start, end, steps - 1) ** 2.0, dim=-1), steps)  # fmt: off

    mask = mask[..., None] * mask[..., None, :]

    output = torch.sum(torch.sum(torch.sum(torch.nn.functional.log_softmax(input, dim=-1) * target, dim=-1) * -1.0 * mask, dim=-1) / (torch.sum(mask, dim=[-1, -2]) + torch.finfo(mask.dtype).eps)[..., None], dim=-1)  # fmt: off

    match reduction:
        case "mean":
            output = torch.mean(output)
        case "sum":
            output = torch.sum(output)

    return output
