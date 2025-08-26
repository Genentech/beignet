import math

import torch
from torch import Tensor


def chi_square_independence_sample_size(
    effect_size: Tensor,
    rows: Tensor,
    cols: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))

    rows = torch.atleast_1d(torch.as_tensor(rows))
    cols = torch.atleast_1d(torch.as_tensor(cols))

    if (
        effect_size.dtype == torch.float64
        or rows.dtype == torch.float64
        or cols.dtype == torch.float64
    ):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)

    rows = rows.to(dtype)
    cols = cols.to(dtype)

    effect_size = torch.clamp(effect_size, min=1e-6)

    rows = torch.clamp(rows, min=2.0)
    cols = torch.clamp(cols, min=2.0)

    degrees_of_freedom = (rows - 1) * (cols - 1)

    sqrt_2 = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2

    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt_2

    n_initial = ((z_alpha + z_beta) / effect_size) ** 2

    min_sample_size = 5.0 * rows * cols

    n_initial = torch.clamp(n_initial, min=min_sample_size)

    n_current = n_initial

    convergence_tolerance = 1e-6

    max_iterations = 10

    for _iteration in range(max_iterations):
        ncp_current = n_current * effect_size**2

        chi2_critical = degrees_of_freedom + z_alpha * torch.sqrt(
            2 * degrees_of_freedom
        )

        mean_nc_chi2 = degrees_of_freedom + ncp_current

        var_nc_chi2 = 2 * (degrees_of_freedom + 2 * ncp_current)

        std_nc_chi2 = torch.sqrt(var_nc_chi2)

        z_score = (chi2_critical - mean_nc_chi2) / torch.clamp(std_nc_chi2, min=1e-10)

        power_current = (1 - torch.erf(z_score / sqrt_2)) / 2

        power_current = torch.clamp(power_current, 0.01, 0.99)

        power_diff = power - power_current

        adjustment = (
            power_diff
            * n_current
            / (2 * torch.clamp(power_current * (1 - power_current), min=0.01))
        )

        converged_mask = torch.abs(power_diff) < convergence_tolerance

        adjustment = torch.where(converged_mask, adjustment * 0.1, adjustment)

        n_current = n_current + adjustment

        n_current = torch.clamp(n_current, min=min_sample_size)

        n_current = torch.clamp(n_current, max=1000000.0)

    output = torch.ceil(n_current)

    output = torch.clamp(output, min=min_sample_size)

    if out is not None:
        out.copy_(output)
        return out

    return output
