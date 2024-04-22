import torch
from torch import Tensor


def _shift(a: Tensor, b: Tensor) -> Tensor:
    x = None
    y = None
    z = None

    if len(b) == 2:
        x, y = b

        z = 0
    elif len(b) == 3:
        x, y, z = b

    if x is not None:
        if x < 0:
            a = torch.concatenate(
                [
                    a[1:],
                    a[:1],
                ],
            )
        elif x > 0:
            a = torch.concatenate(
                [
                    a[-1:],
                    a[:-1],
                ],
            )

    if y is not None:
        if y < 0:
            a = torch.concatenate(
                [
                    a[:, 1:],
                    a[:, :1],
                ],
                dim=1,
            )
        elif y > 0:
            a = torch.concatenate(
                [
                    a[:, -1:],
                    a[:, :-1],
                ],
                dim=1,
            )

    if z is not None:
        if z < 0:
            a = torch.concatenate(
                [
                    a[:, :, 1:],
                    a[:, :, :1],
                ],
                dim=2,
            )
        elif z > 0:
            a = torch.concatenate(
                [
                    a[:, :, -1:],
                    a[:, :, :-1],
                ],
                dim=2,
            )

    return a
