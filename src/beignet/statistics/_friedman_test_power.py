import math

import torch
from torch import Tensor

import beignet.distributions


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

    # Get critical value from central chi-squared distribution
    chi2_dist = beignet.distributions.Chi2(degrees_of_freedom)
    chi_squared_critical = chi2_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))

    # Use non-central chi-squared distribution for power calculation
    nc_chi2_dist = beignet.distributions.NonCentralChi2(degrees_of_freedom, lambda_nc)

    power = torch.clamp(
        0.5
        * (
            1
            - torch.erf(
                (chi_squared_critical - nc_chi2_dist.mean)
                / torch.sqrt(torch.clamp(nc_chi2_dist.variance, min=1e-12))
                / math.sqrt(2.0),
            )
        ),
        0.0,
        1.0,
    )

    if out is not None:
        out.copy_(power)

        return out

    return power
