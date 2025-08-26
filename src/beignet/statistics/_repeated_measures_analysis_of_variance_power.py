import math

import torch
from torch import Tensor


def repeated_measures_analysis_of_variance_power(
    effect_size: Tensor,
    n_subjects: Tensor,
    n_timepoints: Tensor,
    epsilon: Tensor = 1.0,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    n_subjects = torch.atleast_1d(torch.as_tensor(n_subjects))
    n_timepoints = torch.atleast_1d(torch.as_tensor(n_timepoints))
    epsilon = torch.atleast_1d(torch.as_tensor(epsilon))

    dtypes = [effect_size.dtype, n_subjects.dtype, n_timepoints.dtype, epsilon.dtype]
    if any(dt == torch.float64 for dt in dtypes):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    n_subjects = n_subjects.to(dtype)
    n_timepoints = n_timepoints.to(dtype)
    epsilon = epsilon.to(dtype)

    effect_size = torch.clamp(effect_size, min=0.0)
    n_subjects = torch.clamp(n_subjects, min=3.0)
    n_timepoints = torch.clamp(n_timepoints, min=2.0)

    epsilon_min = 1.0 / (n_timepoints - 1.0)
    epsilon = torch.maximum(epsilon, epsilon_min)
    epsilon = torch.clamp(epsilon, max=1.0)

    df_time = n_timepoints - 1.0
    df_error = (n_subjects - 1.0) * (n_timepoints - 1.0)

    df_time_corrected = df_time * epsilon
    df_error_corrected = df_error * epsilon
    df_error_corrected = torch.clamp(df_error_corrected, min=1.0)

    lambda_nc = n_subjects * (effect_size**2) * n_timepoints

    sqrt2 = math.sqrt(2.0)
    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
    chi2_critical = df_time_corrected + z_alpha * torch.sqrt(2 * df_time_corrected)
    f_critical = chi2_critical / df_time_corrected

    mean_nc_chi2 = df_time_corrected + lambda_nc
    var_nc_chi2 = 2 * (df_time_corrected + 2 * lambda_nc)
    mean_f = mean_nc_chi2 / df_time_corrected
    var_f = var_nc_chi2 / (df_time_corrected**2)

    var_f = var_f * (
        (df_error_corrected + 2.0) / torch.clamp(df_error_corrected, min=1.0)
    )

    std_f = torch.sqrt(torch.clamp(var_f, min=1e-12))

    z_score = (f_critical - mean_f) / std_f

    power = 0.5 * (1 - torch.erf(z_score / sqrt2))

    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)
        return out
    return power
