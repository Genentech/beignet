import functools

import torch
from torch import Tensor

import beignet.distributions


def analysis_of_covariance_power(
    input: Tensor,
    sample_size: Tensor,
    groups: Tensor,
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

    sample_size : Tensor

    groups : Tensor

    covariate_r2 : Tensor

    n_covariates : Tensor | int, default 1

    alpha : float, default 0.05

    out : Tensor | None

    Returns
    -------
    Tensor
    """
    input = torch.atleast_1d(input)
    sample_size = torch.atleast_1d(sample_size)
    groups = torch.atleast_1d(groups)
    covariate_r2 = torch.atleast_1d(covariate_r2)
    n_covariates = torch.atleast_1d(torch.as_tensor(n_covariates))

    dtype = functools.reduce(
        torch.promote_types,
        [
            input.dtype,
            sample_size.dtype,
            groups.dtype,
            covariate_r2.dtype,
            n_covariates.dtype,
        ],
    )

    input = input.to(dtype)
    sample_size = sample_size.to(dtype)
    groups = groups.to(dtype)
    covariate_r2 = covariate_r2.to(dtype)
    n_covariates = n_covariates.to(dtype)

    input = torch.clamp(input, min=0.0)
    sample_size = torch.clamp(sample_size, min=3.0)
    groups = torch.clamp(groups, min=2.0)
    covariate_r2 = torch.clamp(covariate_r2, min=0.0, max=1 - torch.finfo(dtype).eps)
    n_covariates = torch.clamp(n_covariates, min=0.0)

    # Degrees of freedom - no need to clamp since inputs are already constrained
    df1 = groups - 1.0  # groups >= 2, so df1 >= 1
    df2 = sample_size - groups - n_covariates  # Could be < 1, so keep one clamp
    df2 = torch.clamp(df2, min=1.0)

    f_dist = beignet.distributions.FisherSnedecor(df1, df2)
    f_critical = f_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))

    lambda_nc = (
        sample_size
        * input**2
        / torch.clamp(1.0 - covariate_r2, min=torch.finfo(dtype).eps)
    )

    mean_f = (df1 + lambda_nc) / df1
    variance_f = 2 * (df1 + 2 * lambda_nc) / (df1**2)
    variance_f = variance_f * (df2 + 2.0) / df2  # df2 already >= 1

    z = (f_critical - mean_f) / torch.clamp(
        torch.sqrt(variance_f),
        min=torch.finfo(dtype).eps,
    )

    # Power calculation - CDF is already bounded [0,1], so 1-CDF is also [0,1]
    power = 1 - beignet.distributions.StandardNormal.from_dtype(dtype).cdf(z)
    output = power

    if out is not None:
        out.copy_(output)

        return out

    return output
