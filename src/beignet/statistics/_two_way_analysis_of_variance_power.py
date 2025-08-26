import math

import torch
from torch import Tensor


def two_way_analysis_of_variance_power(
    effect_size: Tensor,
    sample_size_per_cell: Tensor,
    levels_factor_a: Tensor,
    levels_factor_b: Tensor,
    alpha: float = 0.05,
    effect_type: str = "main_a",
    *,
    out: Tensor | None = None,
) -> Tensor:
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))

    sample_size_per_cell = torch.atleast_1d(torch.as_tensor(sample_size_per_cell))

    levels_factor_a = torch.atleast_1d(torch.as_tensor(levels_factor_a))
    levels_factor_b = torch.atleast_1d(torch.as_tensor(levels_factor_b))

    dtypes = [
        effect_size.dtype,
        sample_size_per_cell.dtype,
        levels_factor_a.dtype,
        levels_factor_b.dtype,
    ]
    if any(dt == torch.float64 for dt in dtypes):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)

    sample_size_per_cell = sample_size_per_cell.to(dtype)

    levels_factor_a = levels_factor_a.to(dtype)
    levels_factor_b = levels_factor_b.to(dtype)

    effect_size = torch.clamp(effect_size, min=0.0)

    sample_size_per_cell = torch.clamp(sample_size_per_cell, min=2.0)

    levels_factor_a = torch.clamp(levels_factor_a, min=2.0)
    levels_factor_b = torch.clamp(levels_factor_b, min=2.0)

    total_n = sample_size_per_cell * levels_factor_a * levels_factor_b

    if effect_type == "main_a":
        df_num = levels_factor_a - 1
    elif effect_type == "main_b":
        df_num = levels_factor_b - 1
    elif effect_type == "interaction":
        df_num = (levels_factor_a - 1) * (levels_factor_b - 1)
    else:
        raise ValueError("effect_type must be 'main_a', 'main_b', or 'interaction'")

    df_den = levels_factor_a * levels_factor_b * (sample_size_per_cell - 1)

    df_den = torch.clamp(df_den, min=1.0)

    lambda_nc = total_n * effect_size**2

    sqrt2 = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2

    chi2_critical = df_num + z_alpha * torch.sqrt(2 * df_num)

    f_critical = chi2_critical / df_num

    mean_nc_chi2 = df_num + lambda_nc

    var_nc_chi2 = 2 * (df_num + 2 * lambda_nc)

    mean_f = mean_nc_chi2 / df_num

    var_f = var_nc_chi2 / (df_num**2)

    var_f = var_f * ((df_den + 2) / torch.clamp(df_den, min=1.0))

    std_f = torch.sqrt(torch.clamp(var_f, min=1e-12))

    z_score = (f_critical - mean_f) / std_f

    power = 0.5 * (1 - torch.erf(z_score / sqrt2))

    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)
        return out
    return power
