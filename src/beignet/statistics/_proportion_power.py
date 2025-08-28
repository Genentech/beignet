import torch
from torch import Tensor

import beignet.distributions


def proportion_power(
    p0: Tensor,
    p1: Tensor,
    sample_size: Tensor,
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
    sample_size : Tensor
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

    p0 = torch.atleast_1d(p0)
    p1 = torch.atleast_1d(p1)

    sample_size = torch.atleast_1d(sample_size)

    dtype = torch.promote_types(p0.dtype, p1.dtype)
    dtype = torch.promote_types(dtype, sample_size.dtype)

    p0 = p0.to(dtype)
    p1 = p1.to(dtype)

    sample_size = sample_size.to(dtype)

    epsilon = torch.finfo(dtype).eps

    p0 = torch.clamp(p0, epsilon, 1 - epsilon)
    p1 = torch.clamp(p1, epsilon, 1 - epsilon)

    se_null = torch.sqrt(p0 * (1 - p0) / sample_size)

    se_alt = torch.sqrt(p1 * (1 - p1) / sample_size)

    effect = (p1 - p0) / se_null

    normal_dist = beignet.distributions.StandardNormal.from_dtype(dtype)

    if alternative == "two-sided":
        z_alpha_half = normal_dist.icdf(torch.tensor(1 - alpha / 2, dtype=dtype))

        variance_ratio = se_alt / se_null

        adjusted_effect = effect / variance_ratio

        power = (1 - normal_dist.cdf(z_alpha_half - adjusted_effect)) + (
            1 - normal_dist.cdf(z_alpha_half + adjusted_effect)
        )

    elif alternative == "greater":
        z_alpha = normal_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))

        variance_ratio = se_alt / se_null

        adjusted_effect = effect / variance_ratio

        power = 1 - normal_dist.cdf(z_alpha - adjusted_effect)

    elif alternative == "less":
        z_alpha = normal_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))

        variance_ratio = se_alt / se_null

        adjusted_effect = effect / variance_ratio

        power = normal_dist.cdf(-z_alpha - adjusted_effect)

    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    if out is not None:
        out.copy_(power)

        return out

    return power
