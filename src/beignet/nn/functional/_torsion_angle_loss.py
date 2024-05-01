from typing import Literal

import torch
from torch import Tensor

from beignet.nn.functional import angle_norm_loss


def torsion_angle_loss(
    input: Tensor,
    target: (Tensor, Tensor),
    reduction: Literal["mean", "sum"] | None = "mean",
) -> Tensor:
    """
    Parameters
    ----------
    input : Tensor
        [*, N, 7, 2]

    target : (Tensor, Tensor)
        [*, N, 7, 2], [*, N, 7, 2]

    reduction : str
        "mean" or "sum"
    """
    x = input / torch.unsqueeze(torch.norm(input, dim=-1), dim=-1)

    y, z = target

    loss = torch.minimum(
        torch.norm(x - y, dim=-1) ** 2,
        torch.norm(x - z, dim=-1) ** 2,
    )

    match reduction:
        case "mean":
            loss = torch.mean(
                loss,
                dim=[-1, -2],
            )
        case "sum":
            loss = torch.sum(
                loss,
                dim=[-1, -2],
            )

    return loss + angle_norm_loss(x, reduction)
