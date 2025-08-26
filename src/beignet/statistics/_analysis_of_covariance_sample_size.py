import math

import torch
from torch import Tensor


def analysis_of_covariance_sample_size(
    effect_size: Tensor,
    groups: Tensor,
    covariate_r2: Tensor,
    n_covariates: Tensor | int = 1,
    power: float = 0.8,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    effect_size_f = torch.atleast_1d(torch.as_tensor(effect_size))

    groups = torch.atleast_1d(torch.as_tensor(groups))

    covariate_r_squared = torch.atleast_1d(torch.as_tensor(covariate_r2))

    num_covariates = torch.atleast_1d(torch.as_tensor(n_covariates))

    dtype = torch.float32
    for tensor in (effect_size_f, groups, covariate_r_squared, num_covariates):
        dtype = torch.promote_types(dtype, tensor.dtype)
    effect_size_f = torch.clamp(effect_size_f.to(dtype), min=1e-8)

    groups = torch.clamp(groups.to(dtype), min=2.0)

    covariate_r_squared = torch.clamp(covariate_r_squared.to(dtype), min=0.0, max=1 - torch.finfo(dtype).eps)

    num_covariates = torch.clamp(num_covariates.to(dtype), min=0.0)

    square_root_two = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * square_root_two

    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * square_root_two

    df1 = groups - 1.0

    chi_squared_critical = df1 + z_alpha * torch.sqrt(2 * df1)

    fcrit_over_df1 = chi_squared_critical / df1

    lam0 = ((z_alpha + z_beta) * math.sqrt(2.0)) ** 2

    n0 = lam0 * torch.clamp(1.0 - covariate_r_squared, min=torch.finfo(dtype).eps) / (effect_size_f**2)

    n0 = torch.clamp(n0, min=groups + num_covariates + 2.0)

    n_curr = n0
    for _ in range(8):
        df2 = torch.clamp(n_curr - groups - num_covariates, min=1.0)

        lambda_nc = (
            n_curr
            * effect_size_f**2
            / torch.clamp(1.0 - covariate_r_squared, min=torch.finfo(dtype).eps)
        )

        mean_nc_chi2 = df1 + lambda_nc

        variance_nc_chi_squared = 2 * (df1 + 2 * lambda_nc)

        mean_f = mean_nc_chi2 / df1

        variance_f = variance_nc_chi_squared / (df1**2)

        variance_f = variance_f * ((df2 + 2.0) / torch.clamp(df2, min=1.0))

        standard_deviation_f = torch.sqrt(variance_f)

        z = (fcrit_over_df1 - mean_f + 0.0) / torch.clamp(
            standard_deviation_f,
            min=1e-10,
        )

        power_curr = 0.5 * (1 - torch.erf(z / math.sqrt(2.0)))

        gap = torch.clamp(power - power_curr, min=-0.45, max=0.45)

        n_curr = torch.clamp(
            n_curr * (1.0 + 1.0 * gap),
            min=groups + num_covariates + 2.0,
            max=torch.tensor(1e7, dtype=dtype),
        )

    n_out = torch.ceil(n_curr)
    if out is not None:
        out.copy_(n_out)
        return out
    return n_out
