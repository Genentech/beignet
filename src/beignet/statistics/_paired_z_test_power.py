import functools

import torch
from torch import Tensor

import beignet.distributions


def paired_z_test_power(
    input: Tensor,
    sample_size: Tensor,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    input : Tensor

    sample_size : Tensor

    alpha : float, default 0.05

    alternative : str, default "two-sided"

    out : Tensor | None

    Returns
    -------
    Tensor
    """
    input = torch.atleast_1d(input)
    sample_size = torch.atleast_1d(sample_size)

    dtype = functools.reduce(torch.promote_types, [input.dtype, sample_size.dtype])

    input = input.to(dtype)
    sample_size = sample_size.to(dtype)

    sample_size = torch.clamp(sample_size, min=1.0)

    noncentrality = input * torch.sqrt(sample_size)

    normal_dist = beignet.distributions.StandardNormal.from_dtype(dtype)

    if alternative == "two-sided":
        z_alpha_half = normal_dist.icdf(torch.tensor(1 - alpha / 2, dtype=dtype))
        power = (1 - normal_dist.cdf(z_alpha_half - noncentrality)) + normal_dist.cdf(
            -z_alpha_half - noncentrality,
        )
    elif alternative == "greater":
        power = 1 - normal_dist.cdf(
            normal_dist.icdf(torch.tensor(1 - alpha, dtype=dtype)) - noncentrality,
        )
    else:
        power = normal_dist.cdf(
            -normal_dist.icdf(torch.tensor(1 - alpha, dtype=dtype)) - noncentrality,
        )

    if out is not None:
        out.copy_(power)

        return out

    return power
