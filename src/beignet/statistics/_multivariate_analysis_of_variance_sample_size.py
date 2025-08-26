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

    dtypes = [effect_size.dtype, n_variables.dtype, n_groups.dtype]
    if any(dt == torch.float64 for dt in dtypes):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    n_variables = n_variables.to(dtype)

    n_groups = n_groups.to(dtype)

    effect_size = torch.clamp(effect_size, min=1e-8)

    n_variables = torch.clamp(n_variables, min=1.0)

    n_groups = torch.clamp(n_groups, min=2.0)

    sqrt2 = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2

    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2

    n_init = (
        ((z_alpha + z_beta) / effect_size) ** 2 * n_variables + n_groups + n_variables
    )
    n_init = torch.clamp(n_init, min=n_groups + n_variables + 10)

    n_current = n_init
    for _ in range(15):
        current_power = multivariate_analysis_of_variance_power(
            effect_size, n_current, n_variables, n_groups, alpha=alpha
        )

        power_gap = torch.clamp(power - current_power, min=-0.4, max=0.4)

        adjustment = 1.0 + 1.2 * power_gap

        n_current = torch.clamp(
            n_current * adjustment, min=n_groups + n_variables + 10, max=1e6
        )

    n_out = torch.ceil(n_current)

    if out is not None:
        out.copy_(n_out)
        return out
    return n_out
