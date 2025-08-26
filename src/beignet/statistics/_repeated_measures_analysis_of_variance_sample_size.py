import math

import torch
from torch import Tensor

from ._repeated_measures_analysis_of_variance_power import (
    repeated_measures_analysis_of_variance_power,
)


def repeated_measures_analysis_of_variance_sample_size(
    input: Tensor,
    n_timepoints: Tensor,
    epsilon: Tensor = 1.0,
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
    n_timepoints : Tensor
        Time parameter.
    epsilon : Tensor, default 1.0
        Epsilon parameter.
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

    n_timepoints = torch.atleast_1d(torch.as_tensor(n_timepoints))

    epsilon = torch.atleast_1d(torch.as_tensor(epsilon))

    dtype = torch.float32
    for tensor in (input, n_timepoints, epsilon):
        dtype = torch.promote_types(dtype, tensor.dtype)

    input = input.to(dtype)

    n_timepoints = n_timepoints.to(dtype)

    epsilon = epsilon.to(dtype)

    input = torch.clamp(input, min=1e-8)

    n_timepoints = torch.clamp(n_timepoints, min=2.0)

    epsilon_min = 1.0 / (n_timepoints - 1.0)

    epsilon = torch.maximum(epsilon, epsilon_min)

    epsilon = torch.clamp(epsilon, max=1.0)

    square_root_two = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * square_root_two

    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * square_root_two

    efficiency = n_timepoints * epsilon

    n_initial = ((z_alpha + z_beta) / input) ** 2 / efficiency

    n_initial = torch.clamp(n_initial, min=5.0)

    n_iteration = n_initial
    for _ in range(12):
        current_power = repeated_measures_analysis_of_variance_power(
            input,
            n_iteration,
            n_timepoints,
            epsilon,
            alpha=alpha,
        )

        power_gap = torch.clamp(power - current_power, min=-0.4, max=0.4)

        adjustment = 1.0 + 1.2 * power_gap

        n_iteration = torch.clamp(n_iteration * adjustment, min=5.0, max=1e5)

    n_out = torch.ceil(n_iteration)

    if out is not None:
        out.copy_(n_out)
        return out
    return n_out
