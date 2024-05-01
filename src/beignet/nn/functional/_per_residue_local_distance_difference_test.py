# ruff: noqa: E501

import torch
import torch.nn.functional
from torch import Tensor


def per_residue_local_distance_difference_test(input: Tensor) -> Tensor:
    """
    Parameters
    ----------
    input : Tensor

    Returns
    -------
    output : Tensor
    """
    output = torch.nn.functional.softmax(input, dim=-1)

    step = 1.0 / input.shape[-1]

    bounds = torch.arange(0.5 * step, 1.0, step)

    return torch.sum(output * torch.reshape(bounds, [*[1 for _ in range(len(output.shape[:-1]))], *bounds.shape]), dim=-1) * 100.0  # fmt: off
