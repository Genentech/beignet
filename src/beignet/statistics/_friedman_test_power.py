import math

import torch
from torch import Tensor


def friedman_test_power(
    input: Tensor,
    n_subjects: Tensor,
    n_treatments: Tensor,
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
    n_treatments : Tensor
        N Treatments parameter.
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

    n_treatments = torch.atleast_1d(torch.as_tensor(n_treatments))

    dtype = torch.float32
    for tensor in (input, n_subjects, n_treatments):
        dtype = torch.promote_types(dtype, tensor.dtype)
    input = input.to(dtype)

    n_subjects = n_subjects.to(dtype)

    n_treatments = n_treatments.to(dtype)

    input = torch.clamp(input, min=0.0)

    n_subjects = torch.clamp(n_subjects, min=3.0)

    n_treatments = torch.clamp(n_treatments, min=3.0)

    degrees_of_freedom = n_treatments - 1

    lambda_nc = 12 * n_subjects * input / (n_treatments * (n_treatments + 1))

    square_root_two = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * square_root_two

    chi_squared_critical = degrees_of_freedom + z_alpha * torch.sqrt(
        2 * degrees_of_freedom,
    )

    mean_nc_chi2 = degrees_of_freedom + lambda_nc

    variance_nc_chi_squared = 2 * (degrees_of_freedom + 2 * lambda_nc)

    std_nc_chi2 = torch.sqrt(torch.clamp(variance_nc_chi_squared, min=1e-12))

    z_score = (chi_squared_critical - mean_nc_chi2) / std_nc_chi2

    power = 0.5 * (1 - torch.erf(z_score / square_root_two))

    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)
        return out
    return power
