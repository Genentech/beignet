import math

import torch
from torch import Tensor

from ._multivariate_analysis_of_variance_power import (
    multivariate_analysis_of_variance_power,
)


def multivariate_analysis_of_variance_sample_size(
    input: Tensor,
    n_variables: Tensor,
    n_groups: Tensor,
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
    n_variables : Tensor
        N Variables parameter.
    n_groups : Tensor
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
    n_variables = torch.atleast_1d(torch.as_tensor(n_variables))

    n_groups = torch.atleast_1d(torch.as_tensor(n_groups))

    dtype = torch.float32
    for tensor in (input, n_variables, n_groups):
        dtype = torch.promote_types(dtype, tensor.dtype)

    input = input.to(dtype)
    n_variables = n_variables.to(dtype)

    n_groups = n_groups.to(dtype)

    input = torch.maximum(input, torch.tensor(torch.finfo(dtype).eps, dtype=dtype))

    n_variables = torch.clamp(n_variables, min=1.0)

    n_groups = torch.clamp(n_groups, min=2.0)

    square_root_two = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * square_root_two

    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * square_root_two

    n_initial = ((z_alpha + z_beta) / input) ** 2 * n_variables + n_groups + n_variables
    n_initial = torch.clamp(n_initial, min=n_groups + n_variables + 10)

    n_iteration = n_initial
    for _ in range(15):
        current_power = multivariate_analysis_of_variance_power(
            input,
            n_iteration,
            n_variables,
            n_groups,
            alpha=alpha,
        )

        power_gap = torch.clamp(power - current_power, min=-0.4, max=0.4)

        adjustment = 1.0 + 1.2 * power_gap

        n_iteration = torch.clamp(
            n_iteration * adjustment,
            min=n_groups + n_variables + 10,
            max=1e6,
        )

    n_out = torch.ceil(n_iteration)

    if out is not None:
        out.copy_(n_out)

        return out

    return n_out
