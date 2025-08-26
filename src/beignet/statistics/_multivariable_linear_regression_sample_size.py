import math

import torch
from torch import Tensor

from ._multivariable_linear_regression_power import (
    multivariable_linear_regression_power,
)


def multivariable_linear_regression_sample_size(
    r_squared: Tensor,
    n_predictors: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r_squared = torch.atleast_1d(torch.as_tensor(r_squared))

    n_predictors = torch.atleast_1d(torch.as_tensor(n_predictors))

    if r_squared.dtype == torch.float64 or n_predictors.dtype == torch.float64:
        dtype = torch.float64
    else:
        dtype = torch.float32

    r_squared = r_squared.to(dtype)

    n_predictors = n_predictors.to(dtype)

    r_squared = torch.clamp(r_squared, min=1e-8, max=0.99)

    n_predictors = torch.clamp(n_predictors, min=1.0)

    sqrt2 = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2

    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2

    f_squared = r_squared / (1 - r_squared)

    n_initial = ((z_alpha + z_beta) ** 2) / f_squared + n_predictors + 1

    n_initial = torch.clamp(n_initial, min=n_predictors + 10)

    n_iteration = n_initial
    for _ in range(15):
        current_power = multivariable_linear_regression_power(
            r_squared, n_iteration, n_predictors, alpha=alpha
        )

        power_gap = torch.clamp(power - current_power, min=-0.4, max=0.4)

        adjustment = 1.0 + 1.1 * power_gap

        n_iteration = torch.clamp(
            n_iteration * adjustment,
            min=n_predictors + 10,
            max=torch.tensor(1e6, dtype=dtype),
        )

    n_out = torch.ceil(n_iteration)

    if out is not None:
        out.copy_(n_out)
        return out
    return n_out
