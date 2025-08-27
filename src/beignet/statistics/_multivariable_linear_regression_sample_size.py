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
    r"""

    Parameters
    ----------
    r_squared : Tensor
        Covariate correlation.
    n_predictors : Tensor
        N Predictors parameter.
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

    r_squared = torch.atleast_1d(torch.as_tensor(r_squared))

    n_predictors = torch.atleast_1d(torch.as_tensor(n_predictors))

    dtype = torch.float32
    for tensor in (r_squared, n_predictors):
        dtype = torch.promote_types(dtype, tensor.dtype)

    r_squared = r_squared.to(dtype)

    n_predictors = n_predictors.to(dtype)

    r_squared = torch.clamp(r_squared, min=1e-8, max=0.99)

    n_predictors = torch.clamp(n_predictors, min=1.0)

    n_initial = torch.clamp(
        (
            (
                torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * math.sqrt(2.0)
                + torch.erfinv(torch.tensor(power, dtype=dtype)) * math.sqrt(2.0)
            )
            ** 2
        )
        / (r_squared / (1 - r_squared))
        + n_predictors
        + 1,
        min=n_predictors + 10,
    )

    n_iteration = n_initial
    for _ in range(15):
        n_iteration = torch.clamp(
            n_iteration
            * (
                1.0
                + 1.1
                * torch.clamp(
                    power
                    - multivariable_linear_regression_power(
                        r_squared,
                        n_iteration,
                        n_predictors,
                        alpha=alpha,
                    ),
                    min=-0.4,
                    max=0.4,
                )
            ),
            min=n_predictors + 10,
            max=torch.tensor(1e6, dtype=dtype),
        )

    n_out = torch.ceil(n_iteration)

    if out is not None:
        out.copy_(n_out)

        return out

    return n_out
