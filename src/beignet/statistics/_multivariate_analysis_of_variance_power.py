import math

import torch
from torch import Tensor

import beignet.distributions


def multivariate_analysis_of_variance_power(
    input: Tensor,
    sample_size: Tensor,
    n_variables: Tensor,
    n_groups: Tensor,
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
    n_variables : Tensor
        N Variables parameter.
    n_groups : Tensor
        Number of groups.
    alpha : float, default 0.05
        Type I error rate.
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        Statistical power.
    """

    input = torch.atleast_1d(torch.as_tensor(input))
    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))
    n_variables = torch.atleast_1d(torch.as_tensor(n_variables))

    n_groups = torch.atleast_1d(torch.as_tensor(n_groups))

    dtype = torch.float32
    for tensor in (input, sample_size, n_variables, n_groups):
        dtype = torch.promote_types(dtype, tensor.dtype)

    input = input.to(dtype)
    sample_size = sample_size.to(dtype)
    n_variables = n_variables.to(dtype)

    n_groups = n_groups.to(dtype)

    input = torch.clamp(input, min=0.0)

    sample_size = torch.clamp(sample_size, min=n_groups + n_variables + 5)

    n_variables = torch.clamp(n_variables, min=1.0)

    n_groups = torch.clamp(n_groups, min=2.0)

    df_hypothesis = n_groups - 1

    df_error = sample_size - n_groups

    df1 = df_hypothesis * n_variables

    df2 = df_error * n_variables - (n_variables - df_hypothesis + 1) / 2

    df2 = torch.clamp(df2, min=1.0)

    effect_size_f_squared = input**2

    lambda_nc = sample_size * effect_size_f_squared

    square_root_two = math.sqrt(2.0)

    f_dist = beignet.distributions.FisherSnedecor(df1, df2)
    f_critical = f_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))

    mean_nc_f = (1.0 + lambda_nc / df1) * (df2 / (df2 - 2.0))

    var_nc_f = (
        2.0 * (df2 / (df2 - 2.0)) ** 2 * ((df1 + lambda_nc) / df1 + (df2 - 2.0) / df2)
    )
    std_nc_f = torch.sqrt(torch.clamp(var_nc_f, min=1e-12))

    z_score = (f_critical - mean_nc_f) / std_nc_f

    power = 0.5 * (1 - torch.erf(z_score / square_root_two))

    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)

        return out

    return power
