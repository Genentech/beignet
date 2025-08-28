import torch
from torch import Tensor

import beignet.distributions


def proportion_two_sample_power(
    p1: Tensor,
    p2: Tensor,
    n1: Tensor,
    n2: Tensor | None = None,
    alpha: float = 0.05,
    alternative: str = "two-sided",
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
    n1 : Tensor
        N1 parameter.
    n2 : Tensor | None, optional
        N2 parameter.
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

    p1 = torch.atleast_1d(p1)
    p2 = torch.atleast_1d(p2)

    sample_size_group_1 = torch.atleast_1d(n1)

    if n2 is None:
        sample_size_group_2 = sample_size_group_1
    else:
        sample_size_group_2 = torch.atleast_1d(n2)

    dtype = torch.promote_types(p1.dtype, p2.dtype)
    dtype = torch.promote_types(dtype, sample_size_group_1.dtype)
    dtype = torch.promote_types(dtype, sample_size_group_2.dtype)

    p1 = p1.to(dtype)
    p2 = p2.to(dtype)

    sample_size_group_1 = sample_size_group_1.to(dtype)
    sample_size_group_2 = sample_size_group_2.to(dtype)

    epsilon = torch.finfo(dtype).eps

    p1 = torch.clamp(p1, epsilon, 1 - epsilon)
    p2 = torch.clamp(p2, epsilon, 1 - epsilon)

    p_pooled = (sample_size_group_1 * p1 + sample_size_group_2 * p2) / (
        sample_size_group_1 + sample_size_group_2
    )
    p_pooled = torch.clamp(p_pooled, epsilon, 1 - epsilon)

    se_null = torch.sqrt(
        p_pooled * (1 - p_pooled) * (1 / sample_size_group_1 + 1 / sample_size_group_2),
    )

    se_alt = torch.sqrt(
        p1 * (1 - p1) / sample_size_group_1 + p2 * (1 - p2) / sample_size_group_2,
    )

    effect = (p1 - p2) / se_null

    effect = p1 - p2

    normal_dist = beignet.distributions.StandardNormal.from_dtype(dtype)

    if alternative == "two-sided":
        z_alpha = normal_dist.icdf(torch.tensor(1 - alpha / 2, dtype=dtype))

        standardized_effect = torch.abs(effect) / se_alt

        power = (1 - normal_dist.cdf(z_alpha - standardized_effect)) + (
            1 - normal_dist.cdf(z_alpha + standardized_effect)
        )

    elif alternative == "greater":
        z_alpha = normal_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))

        standardized_effect = effect / se_alt

        power = 1 - normal_dist.cdf(z_alpha - standardized_effect)

    elif alternative == "less":
        z_alpha = normal_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))

        standardized_effect = effect / se_alt

        power = normal_dist.cdf(-z_alpha - standardized_effect)

    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    if out is not None:
        out.copy_(power)

        return out

    return power
