import math

import torch
from torch import Tensor


def anova_power(
    effect_size: Tensor,
    sample_size: Tensor,
    groups: Tensor,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))

    groups = torch.atleast_1d(torch.as_tensor(groups))

    if (
        effect_size.dtype == torch.float64
        or sample_size.dtype == torch.float64
        or groups.dtype == torch.float64
    ):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    sample_size = sample_size.to(dtype)

    groups = groups.to(dtype)

    effect_size = torch.clamp(effect_size, min=0.0)

    degrees_of_freedom_1 = groups - 1

    degrees_of_freedom_2 = sample_size - groups

    degrees_of_freedom_1 = torch.clamp(degrees_of_freedom_1, min=1.0)
    degrees_of_freedom_2 = torch.clamp(degrees_of_freedom_2, min=1.0)

    sqrt_2 = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2

    chi2_critical = degrees_of_freedom_1 + z_alpha * torch.sqrt(
        2 * degrees_of_freedom_1
    )

    f_critical = chi2_critical / degrees_of_freedom_1

    lambda_nc = sample_size * effect_size**2

    mean_nc_chi2 = degrees_of_freedom_1 + lambda_nc

    var_nc_chi2 = 2 * (degrees_of_freedom_1 + 2 * lambda_nc)

    mean_f = mean_nc_chi2 / degrees_of_freedom_1

    var_f = var_nc_chi2 / (degrees_of_freedom_1**2)

    adjustment_factor = (degrees_of_freedom_2 + 2) / torch.clamp(
        degrees_of_freedom_2, min=1.0
    )
    var_f = var_f * adjustment_factor

    std_f = torch.sqrt(var_f)

    z_score = (f_critical - mean_f) / torch.clamp(std_f, min=1e-10)

    power = (1 - torch.erf(z_score / sqrt_2)) / 2

    output = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
