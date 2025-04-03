from typing import Tuple

import torch
from torch import Tensor


def torsion_angle_loss(input, target: Tuple[Tensor, Tensor]) -> Tensor:
    """

    Parameters
    ----------
    input
    target

    Returns
    -------

    """
    a = input / torch.unsqueeze(torch.norm(input, dim=-1), dim=-1)

    b, c = target

    x = torch.mean(
        torch.minimum(
            torch.square(torch.norm(a - b, dim=-1)),
            torch.square(torch.norm(a - c, dim=-1)),
        ),
        dim=[-1, -2],
    )

    y = torch.mean(
        torch.abs(torch.norm(input, dim=-1) - 1),
        dim=[-1, -2],
    )

    return x + 0.02 * y
