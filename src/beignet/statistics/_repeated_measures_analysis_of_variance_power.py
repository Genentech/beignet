import math

import torch
from torch import Tensor

import beignet.distributions


def repeated_measures_analysis_of_variance_power(
    input: Tensor,
    n_subjects: Tensor,
    n_timepoints: Tensor,
    epsilon: Tensor = 1.0,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    input : Tensor
        Input tensor.
    n_subjects : Tensor
        N Subjects parameter.
    n_timepoints : Tensor
        Time parameter.
    epsilon : Tensor, default 1.0
        Epsilon parameter.
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

    n_subjects = torch.atleast_1d(torch.as_tensor(n_subjects))

    n_timepoints = torch.atleast_1d(torch.as_tensor(n_timepoints))

    epsilon = torch.atleast_1d(torch.as_tensor(epsilon))

    dtype = torch.float32
    for tensor in (input, n_subjects, n_timepoints, epsilon):
        dtype = torch.promote_types(dtype, tensor.dtype)

    input = input.to(dtype)

    n_subjects = n_subjects.to(dtype)

    n_timepoints = n_timepoints.to(dtype)

    epsilon = epsilon.to(dtype)

    input = torch.clamp(input, min=0.0)

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

    lambda_nc = n_subjects * (input**2) * n_timepoints

    square_root_two = math.sqrt(2.0)

    f_dist = beignet.distributions.FisherSnedecor(df_time_corrected, df_error)
    f_critical = f_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))

    mean_nc_chi2 = df_time_corrected + lambda_nc

    variance_nc_chi_squared = 2 * (df_time_corrected + 2 * lambda_nc)

    mean_f = mean_nc_chi2 / df_time_corrected

    variance_f = variance_nc_chi_squared / (df_time_corrected**2)

    variance_f = variance_f * (
        (df_error_corrected + 2.0) / torch.clamp(df_error_corrected, min=1.0)
    )

    standard_deviation_f = torch.sqrt(torch.clamp(variance_f, min=1e-12))

    z_score = (f_critical - mean_f) / standard_deviation_f

    power = 0.5 * (1 - torch.erf(z_score / square_root_two))

    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)
        return out
    return power
