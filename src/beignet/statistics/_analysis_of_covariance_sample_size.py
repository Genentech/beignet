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
    r"""
    Required total sample size for fixed-effects ANCOVA (one-way).

    Parameters
    ----------
    effect_size : Tensor
        Cohen's f for the group effect (pre-adjustment).
    groups : Tensor
        Number of groups.
    covariate_r2 : Tensor
        Proportion of variance explained by covariates (R² in [0,1)).
    n_covariates : Tensor or int, default=1
        Number of covariates.
    power : float, default=0.8
        Target power.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    Tensor
        Required total N (rounded up).
    """
    effect_size_f = torch.atleast_1d(torch.as_tensor(effect_size))
    groups = torch.atleast_1d(torch.as_tensor(groups))
    R2 = torch.atleast_1d(torch.as_tensor(covariate_r2))
    num_covariates = torch.atleast_1d(torch.as_tensor(n_covariates))

    dtype = (
        torch.float64
        if any(
            t.dtype == torch.float64
            for t in (effect_size_f, groups, R2, num_covariates)
        )
        else torch.float32
    )
    effect_size_f = torch.clamp(effect_size_f.to(dtype), min=1e-8)
    groups = torch.clamp(groups.to(dtype), min=2.0)
    R2 = torch.clamp(R2.to(dtype), min=0.0, max=1 - torch.finfo(dtype).eps)
    num_covariates = torch.clamp(num_covariates.to(dtype), min=0.0)

    sqrt2 = math.sqrt(2.0)
    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2

    df1 = groups - 1.0
    chi2_crit = df1 + z_alpha * torch.sqrt(2 * df1)
    fcrit_over_df1 = chi2_crit / df1

    lam0 = ((z_alpha + z_beta) * math.sqrt(2.0)) ** 2
    N0 = lam0 * torch.clamp(1.0 - R2, min=torch.finfo(dtype).eps) / (effect_size_f**2)
    N0 = torch.clamp(N0, min=groups + num_covariates + 2.0)

    N_curr = N0
    for _ in range(8):
        df2 = torch.clamp(N_curr - groups - num_covariates, min=1.0)
        lambda_nc = (
            N_curr
            * effect_size_f**2
            / torch.clamp(1.0 - R2, min=torch.finfo(dtype).eps)
        )

        mean_nc_chi2 = df1 + lambda_nc
        var_nc_chi2 = 2 * (df1 + 2 * lambda_nc)
        mean_f = mean_nc_chi2 / df1
        var_f = var_nc_chi2 / (df1**2)
        var_f = var_f * ((df2 + 2.0) / torch.clamp(df2, min=1.0))

        std_f = torch.sqrt(var_f)
        z = (fcrit_over_df1 - mean_f + 0.0) / torch.clamp(std_f, min=1e-10)
        power_curr = 0.5 * (1 - torch.erf(z / math.sqrt(2.0)))

        gap = torch.clamp(power - power_curr, min=-0.45, max=0.45)
        N_curr = torch.clamp(
            N_curr * (1.0 + 1.0 * gap),
            min=groups + num_covariates + 2.0,
            max=torch.tensor(1e7, dtype=dtype),
        )

    N_out = torch.ceil(N_curr)
    if out is not None:
        out.copy_(N_out)
        return out
    return N_out
