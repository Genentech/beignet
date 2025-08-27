import math

import torch
from torch import Tensor

import beignet.distributions


def mcnemars_test_power(
    p01: Tensor,
    p10: Tensor,
    sample_size: Tensor,
    alpha: float = 0.05,
    two_sided: bool = True,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r""" """
    p01 = torch.atleast_1d(torch.as_tensor(p01))
    p10 = torch.atleast_1d(torch.as_tensor(p10))

    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))

    dtype = (
        torch.float64
        if any(t.dtype == torch.float64 for t in (p01, p10, sample_size))
        else torch.float32
    )
    p01 = torch.clamp(p01.to(dtype), 0.0, 1.0)
    p10 = torch.clamp(p10.to(dtype), 0.0, 1.0)

    sample_size = torch.clamp(sample_size.to(dtype), min=1.0)

    d = sample_size * (p01 + p10)

    probability = torch.where(
        (p01 + p10) > 0,
        p01 / torch.clamp(p01 + p10, min=1e-12),
        torch.zeros_like(p01),
    )
    mean = d * (probability - 0.5)

    standard_deviation = torch.sqrt(torch.clamp(d * 0.25, min=1e-12))

    sqrt2 = math.sqrt(2.0)
    normal_dist = beignet.distributions.StandardNormal.from_dtype(dtype)

    if two_sided:
        z_critical = normal_dist.icdf(torch.tensor(1 - alpha / 2, dtype=dtype))

        z_upper = z_critical - mean / torch.clamp(standard_deviation, min=1e-12)

        z_lower = -z_critical - mean / torch.clamp(standard_deviation, min=1e-12)

        power = (1 - normal_dist.cdf(z_upper)) + normal_dist.cdf(z_lower)
    else:
        z_critical = normal_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))

        zscore = z_critical - mean / torch.clamp(standard_deviation, min=1e-12)

        power = 1 - normal_dist.cdf(zscore)

    out_t = torch.clamp(power, 0.0, 1.0)
    if out is not None:
        out.copy_(out_t)
        return out
    return out_t
