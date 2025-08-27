import torch
from torch import Tensor

import beignet.distributions


def proportion_sample_size(
    p0: Tensor,
    p1: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    p0 : Tensor
        P0 parameter.
    p1 : Tensor
        P1 parameter.
    power : float, default 0.8
        Statistical power.
    alpha : float, default 0.05
        Type I error rate.
    alternative : str, default 'two-sided'
        Alternative hypothesis ("two-sided", "greater", "less").
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        Sample size.
    """

    p0 = torch.atleast_1d(torch.as_tensor(p0))
    p1 = torch.atleast_1d(torch.as_tensor(p1))

    dtype = torch.float32
    for tensor in (p0, p1):
        dtype = torch.promote_types(dtype, tensor.dtype)

    p0 = p0.to(dtype)
    p1 = p1.to(dtype)

    epsilon = 1e-8

    p0 = torch.clamp(p0, epsilon, 1 - epsilon)
    p1 = torch.clamp(p1, epsilon, 1 - epsilon)

    normal_dist = beignet.distributions.Normal(
        torch.tensor(0.0, dtype=dtype),
        torch.tensor(1.0, dtype=dtype),
    )

    if alternative == "two-sided":
        z_alpha = normal_dist.icdf(torch.tensor(1 - alpha / 2, dtype=dtype))
        z_beta = normal_dist.icdf(torch.tensor(power, dtype=dtype))
    elif alternative in ["greater", "less"]:
        z_alpha = normal_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))
        z_beta = normal_dist.icdf(torch.tensor(power, dtype=dtype))
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    result = torch.clamp(
        torch.ceil(
            (
                (
                    z_alpha * torch.sqrt(p0 * (1 - p0))
                    + z_beta * torch.sqrt(p1 * (1 - p1))
                )
                / torch.where(
                    torch.abs(p1 - p0) < 1e-6,
                    torch.tensor(1e-6, dtype=dtype),
                    torch.abs(p1 - p0),
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
