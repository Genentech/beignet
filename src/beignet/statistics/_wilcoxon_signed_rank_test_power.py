import functools

import torch
from torch import Tensor

import beignet.distributions


def wilcoxon_signed_rank_test_power(
    prob_positive: Tensor,
    nobs: Tensor,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r""" """
    probability = torch.atleast_1d(prob_positive)
    sample_size = torch.atleast_1d(nobs)

    dtype = functools.reduce(
        torch.promote_types,
        [probability.dtype, sample_size.dtype],
    )
    probability = probability.to(dtype)

    sample_size = torch.clamp(sample_size.to(dtype), min=5.0)

    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError(
            f"alternative must be 'two-sided', 'greater', or 'less', got {alternative}",
        )

    s = sample_size * (sample_size + 1.0) / 2.0

    noncentrality = (s * probability - s / 2.0) / torch.sqrt(
        torch.clamp(
            sample_size * (sample_size + 1.0) * (2.0 * sample_size + 1.0) / 24.0,
            min=torch.finfo(dtype).eps,
        ),
    )

    normal_dist = beignet.distributions.StandardNormal.from_dtype(dtype)

    if alternative == "two-sided":
        z_critical = normal_dist.icdf(torch.tensor(1 - alpha / 2, dtype=dtype))

        power = (
            1
            - normal_dist.cdf(z_critical - noncentrality)
            + normal_dist.cdf(-z_critical - noncentrality)
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
