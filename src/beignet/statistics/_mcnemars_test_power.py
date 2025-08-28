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
    p01 = torch.atleast_1d(p01)
    p10 = torch.atleast_1d(p10)
    sample_size = torch.atleast_1d(sample_size)

    dtype = torch.promote_types(p01.dtype, p10.dtype)
    dtype = torch.promote_types(dtype, sample_size.dtype)

    p01 = torch.clamp(p01.to(dtype), 0.0, 1.0)
    p10 = torch.clamp(p10.to(dtype), 0.0, 1.0)
    sample_size = torch.clamp(sample_size.to(dtype), min=1.0)

    # Compute shared expressions once
    p_sum = p01 + p10
    p_ratio = torch.where(p_sum > 0, p01 / p_sum, torch.zeros_like(p01))
    effect = p_ratio - 0.5
    variance = torch.clamp(sample_size * p_sum * 0.25, min=torch.finfo(dtype).eps)
    std_error = torch.sqrt(variance)

    # Compute z-statistic
    z_stat = sample_size * p_sum * effect / std_error

    normal_dist = beignet.distributions.StandardNormal.from_dtype(dtype)

    if two_sided:
        z_critical = normal_dist.icdf(torch.tensor(1 - alpha / 2, dtype=dtype))
        power = (
            1
            - normal_dist.cdf(z_critical - z_stat)
            + normal_dist.cdf(-z_critical - z_stat)
        )
    else:
        z_critical = normal_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))
        power = 1 - normal_dist.cdf(z_critical - z_stat)

    if out is not None:
        out.copy_(power)
        return out

    return power
