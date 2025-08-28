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

    input = torch.clamp(input, min=torch.finfo(dtype).eps)

    n_timepoints = torch.clamp(n_timepoints, min=2.0)

    epsilon = torch.maximum(epsilon, 1.0 / (n_timepoints - 1.0))

    epsilon = torch.clamp(epsilon, max=1.0)

    n_iteration = torch.clamp(
        (
            (
                torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * math.sqrt(2.0)
                + torch.erfinv(torch.tensor(power, dtype=dtype)) * math.sqrt(2.0)
            )
            / input
        )
        ** 2
        / (n_timepoints * epsilon),
        min=5.0,
    )

    for _ in range(12):
        n_iteration = torch.clamp(
            n_iteration
            * (
                1.0
                + 1.2
                * torch.clamp(
                    power
                    - repeated_measures_analysis_of_variance_power(
                        input,
                        n_iteration,
                        n_timepoints,
                        epsilon,
                        alpha=alpha,
                    ),
                    min=-0.4,
                    max=0.4,
                )
            ),
            min=5.0,
            max=1e5,
        )

    n_out = torch.ceil(n_iteration)

    if out is not None:
        out.copy_(n_out)

        return out

    return n_out
