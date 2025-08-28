import math

import torch
from torch import Tensor

import beignet.distributions


def chi_square_goodness_of_fit_sample_size(
    input: Tensor,
    degrees_of_freedom: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    input : Tensor
        Input tensor.
    degrees_of_freedom : Tensor
        Degrees of freedom.
    power : float, default 0.8
        Statistical power.
    alpha : float, default 0.05
        Type I error rate.
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        Sample size.
    """

    input = torch.atleast_1d(torch.as_tensor(input))

    degrees_of_freedom = torch.atleast_1d(torch.as_tensor(degrees_of_freedom))

    dtype = torch.float32
    for tensor in (input, degrees_of_freedom):
        dtype = torch.promote_types(dtype, tensor.dtype)

    input = input.to(dtype)

    degrees_of_freedom = degrees_of_freedom.to(dtype)

    input = torch.maximum(input, torch.finfo(dtype).eps)

    degrees_of_freedom = torch.clamp(degrees_of_freedom, min=1.0)

    square_root_two = math.sqrt(2.0)

    chi2_dist = beignet.distributions.Chi2(degrees_of_freedom)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * square_root_two
    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * square_root_two

    n_initial = ((z_alpha + z_beta) / input) ** 2

    n_initial = torch.clamp(n_initial, min=5.0)

    n_iteration = n_initial

    convergence_tolerance = 1e-6

    max_iterations = 10

    for _iteration in range(max_iterations):
        ncp_iteration = n_iteration * input**2

        chi_squared_critical = chi2_dist.icdf(torch.tensor(1 - alpha, dtype=dtype))

        mean_nc_chi2 = degrees_of_freedom + ncp_iteration

        variance_nc_chi_squared = 2 * (degrees_of_freedom + 2 * ncp_iteration)

        std_nc_chi2 = torch.sqrt(variance_nc_chi_squared)

        z_score = (chi_squared_critical - mean_nc_chi2) / torch.clamp(
            std_nc_chi2,
            min=torch.finfo(dtype).eps,
        )

        power_iteration = (1 - torch.erf(z_score / square_root_two)) / 2

        power_iteration = torch.clamp(power_iteration, 0.01, 0.99)

        power_diff = power - power_iteration

        adjustment = (
            power_diff
            * n_iteration
            / (2 * torch.clamp(power_iteration * (1 - power_iteration), min=0.01))
        )

        converged_mask = torch.abs(power_diff) < convergence_tolerance

        adjustment = torch.where(converged_mask, adjustment * 0.1, adjustment)

        n_iteration = n_iteration + adjustment

        n_iteration = torch.clamp(n_iteration, min=5.0)

        n_iteration = torch.clamp(n_iteration, max=1000000.0)

    result = torch.ceil(n_iteration)

    result = torch.clamp(result, min=5.0)

    if out is not None:
        out.copy_(result)

        return out

    return result
