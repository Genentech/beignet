import math

import torch
from torch import Tensor

from ._anova_power import anova_power


def anova_sample_size(
    input: Tensor,
    groups: Tensor,
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
    groups : Tensor
        Number of groups.
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

    groups = torch.atleast_1d(torch.as_tensor(groups))

    dtype = torch.float32
    for tensor in (input, groups):
        dtype = torch.promote_types(dtype, tensor.dtype)

    input = input.to(dtype)

    groups = groups.to(dtype)

    input = torch.clamp(input, min=1e-6)

    groups = torch.clamp(groups, min=2.0)

    df1 = groups - 1

    square_root_two = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * square_root_two

    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * square_root_two

    chi_squared_critical = df1 + z_alpha * torch.sqrt(2 * df1)

    lambda_initial = ((z_alpha + z_beta) * square_root_two) ** 2

    n_initial = lambda_initial / (input**2)

    n_initial = torch.clamp(n_initial, min=groups + 1)

    n_iteration = n_initial

    convergence_tolerance = 1e-6

    max_iterations = 8

    for _iteration in range(max_iterations):
        power_iteration = anova_power(
            input,
            n_iteration,
            groups,
            alpha,
        )

        power_diff = power - power_iteration

        adjustment = power_diff * n_iteration * 0.5

        converged_mask = torch.abs(power_diff) < convergence_tolerance

        adjustment = torch.where(converged_mask, adjustment * 0.1, adjustment)

        n_iteration = n_iteration + adjustment

        n_iteration = torch.clamp(n_iteration, min=groups + 1)

        n_iteration = torch.clamp(n_iteration, max=100000.0)

    result = torch.ceil(n_iteration)

    result = torch.clamp(result, min=groups + 1)

    if out is not None:
        out.copy_(result)
        return out
    return result
