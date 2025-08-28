import functools

import torch
from torch import Tensor

import beignet.distributions


def independent_z_test_power(
    input: Tensor,
    sample_size1: Tensor,
    sample_size2: Tensor | None = None,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    input : Tensor
        Input tensor.
    sample_size1 : Tensor
        Sample size.
    sample_size2 : Tensor | None, optional
        Sample size.
    alpha : float, default 0.05
        Type I error rate.
    alternative : str, default 'two-sided'
        Alternative hypothesis ("two-sided", "greater", "less").
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        Statistical power.
    """

    input = torch.atleast_1d(input)

    sample_size1 = torch.atleast_1d(sample_size1)

    if sample_size2 is None:
        sample_size2 = sample_size1
    else:
        sample_size2 = torch.atleast_1d(sample_size2)

    dtype = functools.reduce(
        torch.promote_types,
        [input.dtype, sample_size1.dtype, sample_size2.dtype],
    )

    input = input.to(dtype)

    sample_size1 = sample_size1.to(dtype)
    sample_size2 = sample_size2.to(dtype)

    sample_size1 = torch.clamp(sample_size1, min=1.0)
    sample_size2 = torch.clamp(sample_size2, min=1.0)

    n_eff = (sample_size1 * sample_size2) / (sample_size1 + sample_size2)

    noncentrality = input * torch.sqrt(n_eff)

    normal_dist = beignet.distributions.StandardNormal.from_dtype(dtype)

    if alternative == "two-sided":
        z_alpha_half = normal_dist.icdf(torch.tensor(1 - alpha / 2, dtype=dtype))

        power_upper = 1 - normal_dist.cdf(z_alpha_half - noncentrality)
        power_lower = normal_dist.cdf(-z_alpha_half - noncentrality)

        power = torch.clamp(power_upper + power_lower, 0.0, 1.0)
    elif alternative == "larger":
        z_alpha = normal_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))

        power = 1 - normal_dist.cdf(z_alpha - noncentrality)
    elif alternative == "smaller":
        z_alpha = normal_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))

        power = normal_dist.cdf(-z_alpha + noncentrality)
    else:
        raise ValueError(
            f"alternative must be 'two-sided', 'larger', or 'smaller', got {alternative}",
        )

    if out is not None:
        out.copy_(power)

        return out

    return power
