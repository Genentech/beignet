from typing import Literal

import torch
from torch import Tensor


def angle_norm_loss(
    input,
    reduction: Literal["mean", "sum"] | None = "mean",
) -> Tensor:
    loss = torch.abs(torch.norm(input, dim=-1) - 1)

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

    return 2.0 * loss
