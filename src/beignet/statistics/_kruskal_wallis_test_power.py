import math

import torch
from torch import Tensor


def kruskal_wallis_test_power(
    effect_size: Tensor,
    sample_sizes: Tensor,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    sample_sizes = torch.atleast_1d(torch.as_tensor(sample_sizes))

    if effect_size.dtype.is_floating_point and sample_sizes.dtype.is_floating_point:
        if effect_size.dtype == torch.float64 or sample_sizes.dtype == torch.float64:
            dtype = torch.float64
        else:
            dtype = torch.float32
    else:
        dtype = torch.float32
    effect_size = effect_size.to(dtype)
    sample_sizes = sample_sizes.to(dtype)

    effect_size = torch.clamp(effect_size, min=0.0)
    sample_sizes = torch.clamp(sample_sizes, min=2.0)

    groups = torch.tensor(sample_sizes.shape[-1], dtype=dtype)
    n = torch.sum(sample_sizes, dim=-1)

    degrees_of_freedom = groups - 1

    lambda_nc = 12 * n * effect_size / (n + 1)

    sqrt2 = math.sqrt(2.0)
    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
    chi2_critical = degrees_of_freedom + z_alpha * torch.sqrt(2 * degrees_of_freedom)

    mean_nc_chi2 = degrees_of_freedom + lambda_nc
    var_nc_chi2 = 2 * (degrees_of_freedom + 2 * lambda_nc)
    std_nc_chi2 = torch.sqrt(torch.clamp(var_nc_chi2, min=1e-12))

    z_score = (chi2_critical - mean_nc_chi2) / std_nc_chi2

    power = 0.5 * (1 - torch.erf(z_score / sqrt2))

    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)
        return out
    return power
