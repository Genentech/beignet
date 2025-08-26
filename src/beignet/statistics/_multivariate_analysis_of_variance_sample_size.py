import math

import torch
from torch import Tensor

from ._multivariate_analysis_of_variance_power import (
    multivariate_analysis_of_variance_power,
)


def multivariate_analysis_of_variance_sample_size(
    effect_size: Tensor,
    n_variables: Tensor,
    n_groups: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    n_variables = torch.atleast_1d(torch.as_tensor(n_variables))

    n_groups = torch.atleast_1d(torch.as_tensor(n_groups))

    dtype = torch.float32
    for tensor in (effect_size, n_variables, n_groups):
        dtype = torch.promote_types(dtype, tensor.dtype)

    effect_size = effect_size.to(dtype)
    n_variables = n_variables.to(dtype)

    n_groups = n_groups.to(dtype)

    effect_size = torch.clamp(effect_size, min=1e-8)

    n_variables = torch.clamp(n_variables, min=1.0)

    n_groups = torch.clamp(n_groups, min=2.0)

    square_root_two = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * square_root_two

    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * square_root_two

    n_initial = (
        ((z_alpha + z_beta) / effect_size) ** 2 * n_variables + n_groups + n_variables
    )
    n_initial = torch.clamp(n_initial, min=n_groups + n_variables + 10)

    n_iteration = n_initial
    for _ in range(15):
        current_power = multivariate_analysis_of_variance_power(
            effect_size,
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
