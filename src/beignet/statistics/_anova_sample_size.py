import math

import torch
from torch import Tensor


def anova_sample_size(
    effect_size: Tensor,
    groups: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))

    groups = torch.atleast_1d(torch.as_tensor(groups))

    dtype = torch.float32
    for tensor in (effect_size, groups):
        dtype = torch.promote_types(dtype, tensor.dtype)

    effect_size = effect_size.to(dtype)

    groups = groups.to(dtype)

    effect_size = torch.clamp(effect_size, min=1e-6)

    groups = torch.clamp(groups, min=2.0)

    df1 = groups - 1

    square_root_two = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * square_root_two

    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * square_root_two

    chi_squared_critical = df1 + z_alpha * torch.sqrt(2 * df1)

    lambda_initial = ((z_alpha + z_beta) * square_root_two) ** 2

    n_initial = lambda_initial / (effect_size**2)

    n_initial = torch.clamp(n_initial, min=groups + 1)

    n_iteration = n_initial

    convergence_tolerance = 1e-6

    max_iterations = 8

    for _iteration in range(max_iterations):
        df2_iteration = n_iteration - groups

        df2_iteration = torch.clamp(df2_iteration, min=1.0)

        lambda_iteration = n_iteration * effect_size**2

        f_critical = chi_squared_critical / df1

        adjustment = 1 + 2 / torch.clamp(df2_iteration, min=1.0)

        f_critical = f_critical * adjustment

        mean_nc_chi2 = df1 + lambda_iteration

        variance_nc_chi_squared = 2 * (df1 + 2 * lambda_iteration)

        mean_f = mean_nc_chi2 / df1

        variance_f = variance_nc_chi_squared / (df1**2)

        var_adjustment = (df2_iteration + 2) / torch.clamp(df2_iteration, min=1.0)

        variance_f = variance_f * var_adjustment

        standard_deviation_f = torch.sqrt(variance_f)

        z_iteration = (f_critical - mean_f) / torch.clamp(
            standard_deviation_f,
            min=1e-10,
        )

        power_iteration = (1 - torch.erf(z_iteration / square_root_two)) / 2

        power_diff = power - power_iteration

        adjustment = power_diff * n_iteration * 0.5

        converged_mask = torch.abs(power_diff) < convergence_tolerance

        adjustment = torch.where(converged_mask, adjustment * 0.1, adjustment)

        n_iteration = n_iteration + adjustment

        n_iteration = torch.clamp(n_iteration, min=groups + 1)

        n_iteration = torch.clamp(n_iteration, max=100000.0)

    result = torch.ceil(n_iteration)

    result = torch.clamp(result, min=groups + 1)

    if out is not None:
        out.copy_(result)
        return out

