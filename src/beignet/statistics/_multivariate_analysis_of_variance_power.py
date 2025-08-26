import math

import torch
from torch import Tensor


def multivariate_analysis_of_variance_power(
    effect_size: Tensor,
    sample_size: Tensor,
    n_variables: Tensor,
    n_groups: Tensor,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))
    n_variables = torch.atleast_1d(torch.as_tensor(n_variables))

    n_groups = torch.atleast_1d(torch.as_tensor(n_groups))

    dtypes = [effect_size.dtype, sample_size.dtype, n_variables.dtype, n_groups.dtype]
    if any(dt == torch.float64 for dt in dtypes):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    sample_size = sample_size.to(dtype)
    n_variables = n_variables.to(dtype)

    n_groups = n_groups.to(dtype)

    effect_size = torch.clamp(effect_size, min=0.0)

    sample_size = torch.clamp(sample_size, min=n_groups + n_variables + 5)

    n_variables = torch.clamp(n_variables, min=1.0)

    n_groups = torch.clamp(n_groups, min=2.0)

    df_hypothesis = n_groups - 1

    df_error = sample_size - n_groups

    df1 = df_hypothesis * n_variables

    df2 = df_error * n_variables - (n_variables - df_hypothesis + 1) / 2

    df2 = torch.clamp(df2, min=1.0)

    effect_size_f_squared = effect_size**2

    lambda_nc = sample_size * effect_size_f_squared

    sqrt2 = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2

    chi_squared_critical = df1 + z_alpha * torch.sqrt(2 * df1)

    f_critical = chi_squared_critical / df1

    mean_nc_f = (1.0 + lambda_nc / df1) * (df2 / (df2 - 2.0))

    var_nc_f = (
        2.0 * (df2 / (df2 - 2.0)) ** 2 * ((df1 + lambda_nc) / df1 + (df2 - 2.0) / df2)
    )
    std_nc_f = torch.sqrt(torch.clamp(var_nc_f, min=1e-12))

    z_score = (f_critical - mean_nc_f) / std_nc_f

    power = 0.5 * (1 - torch.erf(z_score / sqrt2))

    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)
        return out
    return power
