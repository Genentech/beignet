import math

import torch
from torch import Tensor


def multivariable_linear_regression_power(
    r_squared: Tensor,
    sample_size: Tensor,
    n_predictors: Tensor,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    r_squared : Tensor
        Covariate correlation.
    sample_size : Tensor
        Sample size.
    n_predictors : Tensor
        N Predictors parameter.
    alpha : float, default 0.05
        Type I error rate.
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        Statistical power.
    """

    r_squared = torch.atleast_1d(torch.as_tensor(r_squared))

    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))

    n_predictors = torch.atleast_1d(torch.as_tensor(n_predictors))

    dtypes = [r_squared.dtype, sample_size.dtype, n_predictors.dtype]
    if any(dt == torch.float64 for dt in dtypes):
        dtype = torch.float64
    else:
        dtype = torch.float32

    r_squared = r_squared.to(dtype)

    sample_size = sample_size.to(dtype)

    n_predictors = n_predictors.to(dtype)

    r_squared = torch.clamp(r_squared, min=0.0, max=0.99)

    sample_size = torch.clamp(sample_size, min=n_predictors + 10)

    n_predictors = torch.clamp(n_predictors, min=1.0)

    power = torch.clamp(
        0.5
        * (
            1
            - torch.erf(
                (
                    1.0
                    + torch.erfinv(torch.tensor(1 - alpha, dtype=dtype))
                    * math.sqrt(2.0)
                    * torch.sqrt(2.0 / n_predictors)
                    - (1 + sample_size * r_squared / (1 - r_squared) / n_predictors)
                    * (
                        torch.clamp(sample_size - n_predictors - 1, min=1.0)
                        / (torch.clamp(sample_size - n_predictors - 1, min=1.0) - 2)
                    )
                )
                / torch.sqrt(
                    torch.clamp(
                        (
                            2
                            * (
                                torch.clamp(sample_size - n_predictors - 1, min=1.0)
                                / (
                                    torch.clamp(sample_size - n_predictors - 1, min=1.0)
                                    - 2
                                )
                            )
                            ** 2
                            * (
                                (
                                    n_predictors
                                    + sample_size * r_squared / (1 - r_squared)
                                )
                                / n_predictors
                                + (
                                    torch.clamp(sample_size - n_predictors - 1, min=1.0)
                                    - 2
                                )
                                / torch.clamp(sample_size - n_predictors - 1, min=1.0)
                            )
                        ),
                        min=1e-12,
                    ),
                )
                / math.sqrt(2.0),
            )
        ),
        0.0,
        1.0,
    )

    if out is not None:
        out.copy_(power)

        return out

    return power
