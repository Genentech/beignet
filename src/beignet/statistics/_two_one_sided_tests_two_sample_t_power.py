import math

import torch
from torch import Tensor

import beignet.distributions


def two_one_sided_tests_two_sample_t_power(
    true_effect: Tensor,
    nobs1: Tensor,
    ratio: Tensor | float | None = None,
    low: Tensor = 0.0,
    high: Tensor = 0.0,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r""" """
    true_effect_size = torch.atleast_1d(torch.as_tensor(true_effect))

    sample_size_group_1 = torch.atleast_1d(torch.as_tensor(nobs1))
    if ratio is None:
        ratio_t = torch.tensor(1.0)
    else:
        ratio_t = torch.atleast_1d(torch.as_tensor(ratio))

    low = torch.atleast_1d(torch.as_tensor(low))

    high = torch.atleast_1d(torch.as_tensor(high))

    dtype = (
        torch.float64
        if any(
            t.dtype == torch.float64
            for t in (true_effect_size, sample_size_group_1, ratio_t, low, high)
        )
        else torch.float32
    )
    true_effect_size = true_effect_size.to(dtype)

    sample_size_group_1 = sample_size_group_1.to(dtype)

    ratio_t = ratio_t.to(dtype)

    low = low.to(dtype)

    high = high.to(dtype)

    sample_size_group_1 = torch.clamp(sample_size_group_1, min=2.0)

    ratio_t = torch.clamp(ratio_t, min=0.1, max=10.0)

    sample_size_group_2 = sample_size_group_1 * ratio_t

    degrees_of_freedom = torch.clamp(
        sample_size_group_1 + sample_size_group_2 - 2,
        min=1.0,
    )

    se_factor = torch.sqrt(1.0 / sample_size_group_1 + 1.0 / sample_size_group_2)

    noncentrality_parameter_low = (true_effect_size - low) / torch.clamp(
        se_factor,
        min=1e-12,
    )
    noncentrality_parameter_high = (true_effect_size - high) / torch.clamp(
        se_factor,
        min=1e-12,
    )

    square_root_two = math.sqrt(2.0)

    t_dist = beignet.distributions.StudentT(degrees_of_freedom)
    t_critical = t_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))

    def power_greater(noncentrality: Tensor) -> Tensor:
        nc_t_dist = beignet.distributions.NonCentralT(degrees_of_freedom, noncentrality)
        mean_nct = nc_t_dist.mean
        variance_nct = nc_t_dist.variance
        standard_deviation = torch.sqrt(variance_nct)

        zscore = (t_critical - mean_nct) / torch.clamp(
            standard_deviation,
            min=1e-10,
        )
        return 0.5 * (
            1 - torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
        )

    def power_less(noncentrality: Tensor) -> Tensor:
        nc_t_dist = beignet.distributions.NonCentralT(degrees_of_freedom, noncentrality)
        mean_nct = nc_t_dist.mean
        variance_nct = nc_t_dist.variance
        standard_deviation = torch.sqrt(variance_nct)

        zscore = (-t_critical - mean_nct) / torch.clamp(
            standard_deviation,
            min=1e-10,
        )
        return 0.5 * (
            1 + torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
        )

    p_lower = power_greater(noncentrality_parameter_low)

    p_upper = power_less(noncentrality_parameter_high)

    power = torch.minimum(p_lower, p_upper)

    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)
        return out
    return power
