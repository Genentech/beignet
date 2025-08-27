import math

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
    probability = torch.atleast_1d(torch.as_tensor(prob_positive))

    sample_size = torch.atleast_1d(torch.as_tensor(nobs))

    dtype = (
        torch.float64
        if (probability.dtype == torch.float64 or sample_size.dtype == torch.float64)
        else torch.float32
    )
    probability = probability.to(dtype)

    sample_size = torch.clamp(sample_size.to(dtype), min=5.0)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    s = sample_size * (sample_size + 1.0) / 2.0

    mean0 = s / 2.0

    var0 = sample_size * (sample_size + 1.0) * (2.0 * sample_size + 1.0) / 24.0

    sd0 = torch.sqrt(torch.clamp(var0, min=1e-12))

    mean1 = s * probability

    noncentrality = (mean1 - mean0) / sd0

    square_root_two = math.sqrt(2.0)

    normal_dist = beignet.distributions.StandardNormal.from_dtype(dtype)

    if alt == "two-sided":
        z_critical = normal_dist.icdf(torch.tensor(1 - alpha / 2, dtype=dtype))

        upper = 1 - normal_dist.cdf(z_critical - noncentrality)

        lower = normal_dist.cdf(-z_critical - noncentrality)

        power = upper + lower
    elif alt == "greater":
        z_critical = normal_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))

        power = 1 - normal_dist.cdf(z_critical - noncentrality)
    else:
        z_critical = normal_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))

        power = normal_dist.cdf(-z_critical - noncentrality)

    out_t = torch.clamp(power, 0.0, 1.0)
    if out is not None:
        out.copy_(out_t)
        return out
    return out_t
