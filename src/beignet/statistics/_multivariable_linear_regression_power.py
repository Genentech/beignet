import math

import torch
from torch import Tensor


def multivariable_linear_regression_power(
    r_squared: Tensor,
    sample_size: Tensor,
    n_predictors: Tensor,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    r_squared : Tensor
        Covariate correlation.
    sample_size : Tensor
        Sample size.
    n_predictors : Tensor
        N Predictors parameter.
    alpha : float, default 0.05
        Type I error rate.
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        Statistical power.
    """

    r_squared = torch.atleast_1d(torch.as_tensor(r_squared))

    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))

    n_predictors = torch.atleast_1d(torch.as_tensor(n_predictors))

    dtypes = [r_squared.dtype, sample_size.dtype, n_predictors.dtype]
    if any(dt == torch.float64 for dt in dtypes):
        dtype = torch.float64
    else:
        dtype = torch.float32

    r_squared = r_squared.to(dtype)

    sample_size = sample_size.to(dtype)

    n_predictors = n_predictors.to(dtype)

    r_squared = torch.clamp(r_squared, min=0.0, max=0.99)

    sample_size = torch.clamp(sample_size, min=n_predictors + 10)

    n_predictors = torch.clamp(n_predictors, min=1.0)

    df_num = n_predictors

    df_den = sample_size - n_predictors - 1

    df_den = torch.clamp(df_den, min=1.0)

    square_root_two = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * square_root_two

    f_critical = 1.0 + z_alpha * torch.sqrt(2.0 / df_num)

    lambda_nc = sample_size * r_squared / (1 - r_squared)

    mean_nf = (1 + lambda_nc / df_num) * (df_den / (df_den - 2))

    var_nf = (
        2
        * (df_den / (df_den - 2)) ** 2
        * ((df_num + lambda_nc) / df_num + (df_den - 2) / df_den)
    )
    std_nf = torch.sqrt(torch.clamp(var_nf, min=1e-12))

    z_score = (f_critical - mean_nf) / std_nf

    power = 0.5 * (1 - torch.erf(z_score / square_root_two))

    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)
        return out
    return power
