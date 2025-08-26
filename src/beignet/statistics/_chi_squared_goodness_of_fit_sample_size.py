import math

import torch
from torch import Tensor


def chi_square_goodness_of_fit_sample_size(
    effect_size: Tensor,
    degrees_of_freedom: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))

    degrees_of_freedom = torch.atleast_1d(torch.as_tensor(degrees_of_freedom))

    dtype = torch.float32
    for tensor in (effect_size, degrees_of_freedom):
        dtype = torch.promote_types(dtype, tensor.dtype)

    effect_size = effect_size.to(dtype)

    degrees_of_freedom = degrees_of_freedom.to(dtype)

    effect_size = torch.clamp(effect_size, min=1e-6)

    degrees_of_freedom = torch.clamp(degrees_of_freedom, min=1.0)

    square_root_two = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * square_root_two

    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * square_root_two

    n_initial = ((z_alpha + z_beta) / effect_size) ** 2

    n_initial = torch.clamp(n_initial, min=5.0)

    n_iteration = n_initial

    convergence_tolerance = 1e-6

    max_iterations = 10

    for _iteration in range(max_iterations):
        ncp_iteration = n_iteration * effect_size**2

        chi_squared_critical = degrees_of_freedom + z_alpha * torch.sqrt(
            2 * degrees_of_freedom,
        )

        mean_nc_chi2 = degrees_of_freedom + ncp_iteration

        variance_nc_chi_squared = 2 * (degrees_of_freedom + 2 * ncp_iteration)

        std_nc_chi2 = torch.sqrt(variance_nc_chi_squared)

        z_score = (chi_squared_critical - mean_nc_chi2) / torch.clamp(
            std_nc_chi2,
            min=1e-10,
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

