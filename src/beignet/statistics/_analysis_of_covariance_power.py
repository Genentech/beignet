import math

import torch
from torch import Tensor


def analysis_of_covariance_power(
    input: Tensor,
    sample_size: Tensor,
    k: Tensor,
    covariate_r2: Tensor,
    n_covariates: Tensor | int = 1,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    input : Tensor
        Input tensor.
    sample_size : Tensor
        Sample size.
    k : Tensor
        Number of groups.
    covariate_r2 : Tensor
        Covariate correlation.
    n_covariates : Tensor | int, default 1
        Covariate correlation.
    alpha : float, default 0.05
        Type I error rate.
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        Statistical power.
    """

    effect_size_f = torch.atleast_1d(torch.as_tensor(input))

    n = torch.atleast_1d(torch.as_tensor(sample_size))

    groups = torch.atleast_1d(torch.as_tensor(k))

    covariate_r_squared = torch.atleast_1d(torch.as_tensor(covariate_r2))

    num_covariates = torch.atleast_1d(torch.as_tensor(n_covariates))

    dtype = torch.float32
    for tensor in (effect_size_f, n, groups, covariate_r_squared, num_covariates):
        dtype = torch.promote_types(dtype, tensor.dtype)
    effect_size_f = torch.clamp(effect_size_f.to(dtype), min=0.0)

    n = torch.clamp(n.to(dtype), min=3.0)

    groups = torch.clamp(groups.to(dtype), min=2.0)

    covariate_r_squared = torch.clamp(
        covariate_r_squared.to(dtype),
        min=0.0,
        max=1 - torch.finfo(dtype).eps,
    )

    num_covariates = torch.clamp(num_covariates.to(dtype), min=0.0)

    df1 = torch.clamp(groups - 1.0, min=1.0)

    df2 = torch.clamp(n - groups - num_covariates, min=1.0)

    square_root_two = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * square_root_two

    chi_squared_critical = df1 + z_alpha * torch.sqrt(2 * df1)

    f_critical = chi_squared_critical / df1

    lambda_nc = (
        n
        * effect_size_f**2
        / torch.clamp(1.0 - covariate_r_squared, min=torch.finfo(dtype).eps)
    )

    mean_nc_chi2 = df1 + lambda_nc

    variance_nc_chi_squared = 2 * (df1 + 2 * lambda_nc)

    mean_f = mean_nc_chi2 / df1

    variance_f = variance_nc_chi_squared / (df1**2)

    variance_f = variance_f * ((df2 + 2.0) / torch.clamp(df2, min=1.0))

    standard_deviation_f = torch.sqrt(variance_f)

    z = (f_critical - mean_f) / torch.clamp(standard_deviation_f, min=1e-10)

    power = 0.5 * (1 - torch.erf(z / square_root_two))

    out_t = torch.clamp(power, 0.0, 1.0)
    if out is not None:
        out.copy_(out_t)
        return out
    return out_t
