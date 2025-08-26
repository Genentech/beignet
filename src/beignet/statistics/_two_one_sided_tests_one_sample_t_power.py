import math

import torch
from torch import Tensor


def two_one_sided_tests_one_sample_t_power(
    true_effect: Tensor,
    sample_size: Tensor,
    low: Tensor,
    high: Tensor,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    true_effect_size = torch.atleast_1d(torch.as_tensor(true_effect))

    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))

    low = torch.atleast_1d(torch.as_tensor(low))

    high = torch.atleast_1d(torch.as_tensor(high))

    dtype = (
        torch.float64
        if any(
            t.dtype == torch.float64 for t in (true_effect_size, sample_size, low, high)
        )
        else torch.float32
    )
    true_effect_size = true_effect_size.to(dtype)

    sample_size = sample_size.to(dtype)

    low = low.to(dtype)

    high = high.to(dtype)

    sample_size = torch.clamp(sample_size, min=2.0)

    degrees_of_freedom = sample_size - 1

    ncp_low = (true_effect_size - low) * torch.sqrt(sample_size)

    ncp_high = (true_effect_size - high) * torch.sqrt(sample_size)

    sqrt2 = math.sqrt(2.0)

    z = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2

    t_critical = z * torch.sqrt(1 + 1 / (2 * degrees_of_freedom))

    def power_greater(noncentrality: Tensor) -> Tensor:
        variance = torch.where(
            degrees_of_freedom > 2,
            (degrees_of_freedom + noncentrality**2) / (degrees_of_freedom - 2),
            1 + noncentrality**2 / (2 * torch.clamp(degrees_of_freedom, min=1.0)),
        )
        standard_deviation = torch.sqrt(variance)

        zscore = (t_critical - noncentrality) / torch.clamp(
            standard_deviation, min=1e-10
        )
        return 0.5 * (
            1 - torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
        )

    def power_less(noncentrality: Tensor) -> Tensor:
        variance = torch.where(
            degrees_of_freedom > 2,
            (degrees_of_freedom + noncentrality**2) / (degrees_of_freedom - 2),
            1 + noncentrality**2 / (2 * torch.clamp(degrees_of_freedom, min=1.0)),
        )
        standard_deviation = torch.sqrt(variance)

        zscore = (-t_critical - noncentrality) / torch.clamp(
            standard_deviation, min=1e-10
        )
        return 0.5 * (
            1 + torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
        )

    p_lower = power_greater(ncp_low)

    p_upper = power_less(ncp_high)

    power = torch.minimum(p_lower, p_upper)

    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)
        return out
    return power
