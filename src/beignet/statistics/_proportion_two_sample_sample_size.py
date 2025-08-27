import math

import torch
from torch import Tensor


def proportion_two_sample_sample_size(
    p1: Tensor,
    p2: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    ratio: float = 1.0,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    p1 : Tensor
        P1 parameter.
    p2 : Tensor
        P2 parameter.
    power : float, default 0.8
        Statistical power.
    alpha : float, default 0.05
        Type I error rate.
    alternative : str, default 'two-sided'
        Alternative hypothesis ("two-sided", "greater", "less").
    ratio : float, default 1.0
        Sample size ratio.
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        Sample size.
    """

    p1 = torch.atleast_1d(torch.as_tensor(p1))
    p2 = torch.atleast_1d(torch.as_tensor(p2))

    dtype = torch.float32
    for tensor in (p1, p2):
        dtype = torch.promote_types(dtype, tensor.dtype)

    p1 = p1.to(dtype)
    p2 = p2.to(dtype)

    ratio = torch.tensor(ratio, dtype=dtype)

    p1 = torch.clamp(p1, 1e-8, 1 - 1e-8)
    p2 = torch.clamp(p2, 1e-8, 1 - 1e-8)

    if alternative == "two-sided":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * math.sqrt(
            2.0,
        )

        z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * math.sqrt(2.0)
    elif alternative in ["greater", "less"]:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * math.sqrt(2.0)

        z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * math.sqrt(2.0)
        raise ValueError(f"Unknown alternative: {alternative}")

    result = torch.clamp(
        torch.ceil(
            (
                (
                    z_alpha
                    * torch.sqrt(
                        torch.clamp((p1 + p2 * ratio) / (1 + ratio), 1e-8, 1 - 1e-8)
                        * (
                            1
                            - torch.clamp(
                                (p1 + p2 * ratio) / (1 + ratio),
                                1e-8,
                                1 - 1e-8,
                            )
                        )
                        * (1 + 1 / ratio),
                    )
                    + z_beta * torch.sqrt(p1 * (1 - p1) + p2 * (1 - p2) / ratio)
                )
                / torch.where(
                    torch.abs(p1 - p2) < 1e-6,
                    torch.tensor(1e-6, dtype=dtype),
                    torch.abs(p1 - p2),
                )
            )
            ** 2,
        ),
        min=1.0,
    )

    if out is not None:
        out.copy_(result)

        return out

    return result
