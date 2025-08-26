import math

import torch
from torch import Tensor


def analysis_of_covariance_power(
    effect_size: Tensor,
    sample_size: Tensor,
    k: Tensor,
    covariate_r2: Tensor,
    n_covariates: Tensor | int = 1,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    effect_size_f = torch.atleast_1d(torch.as_tensor(effect_size))

    n = torch.atleast_1d(torch.as_tensor(sample_size))

    groups = torch.atleast_1d(torch.as_tensor(k))

    r2 = torch.atleast_1d(torch.as_tensor(covariate_r2))

    num_covariates = torch.atleast_1d(torch.as_tensor(n_covariates))

    dtype = (
        torch.float64
        if any(
            t.dtype == torch.float64
            for t in (effect_size_f, n, groups, r2, num_covariates)
        )
        else torch.float32
    )
    effect_size_f = torch.clamp(effect_size_f.to(dtype), min=0.0)

    n = torch.clamp(n.to(dtype), min=3.0)

    groups = torch.clamp(groups.to(dtype), min=2.0)

    r2 = torch.clamp(r2.to(dtype), min=0.0, max=1 - torch.finfo(dtype).eps)

    num_covariates = torch.clamp(num_covariates.to(dtype), min=0.0)

    df1 = torch.clamp(groups - 1.0, min=1.0)

    df2 = torch.clamp(n - groups - num_covariates, min=1.0)

    sqrt2 = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2

    chi2_crit = df1 + z_alpha * torch.sqrt(2 * df1)

    f_crit = chi2_crit / df1

    lambda_nc = n * effect_size_f**2 / torch.clamp(1.0 - r2, min=torch.finfo(dtype).eps)

    mean_nc_chi2 = df1 + lambda_nc

    var_nc_chi2 = 2 * (df1 + 2 * lambda_nc)

    mean_f = mean_nc_chi2 / df1

    var_f = var_nc_chi2 / (df1**2)

    var_f = var_f * ((df2 + 2.0) / torch.clamp(df2, min=1.0))

    std_f = torch.sqrt(var_f)

    z = (f_crit - mean_f) / torch.clamp(std_f, min=1e-10)

    power = 0.5 * (1 - torch.erf(z / sqrt2))

    out_t = torch.clamp(power, 0.0, 1.0)
    if out is not None:
        out.copy_(out_t)
        return out
    return out_t
