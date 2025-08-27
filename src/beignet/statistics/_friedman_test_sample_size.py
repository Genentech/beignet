import math

import torch
from torch import Tensor

from ._friedman_test_power import friedman_test_power


def friedman_test_sample_size(
    input: Tensor,
    n_treatments: Tensor,
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
    n_treatments : Tensor
        N Treatments parameter.
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

    n_treatments = torch.atleast_1d(torch.as_tensor(n_treatments))

    dtype = (
        torch.float64
        if (input.dtype == torch.float64 or n_treatments.dtype == torch.float64)
        else torch.float32
    )
    input = input.to(dtype)

    n_treatments = n_treatments.to(dtype)

    input = torch.clamp(input, min=1e-8)

    n_treatments = torch.clamp(n_treatments, min=3.0)

    n_initial = torch.clamp(
        (
            (
                (
                    torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * math.sqrt(2.0)
                    + torch.erfinv(torch.tensor(power, dtype=dtype)) * math.sqrt(2.0)
                )
                ** 2
            )
            * n_treatments
            * (n_treatments + 1)
            / (12 * input)
        ),
        min=5.0,
    )

    n_iteration = n_initial

    for _ in range(12):
        n_iteration = torch.clamp(
            n_iteration
            * (
                1.0
                + 1.2
                * torch.clamp(
                    power
                    - friedman_test_power(
                        input,
                        n_iteration,
                        n_treatments,
                        alpha=alpha,
                    ),
                    min=-0.4,
                    max=0.4,
                )
            ),
            min=5.0,
            max=1e6,
        )

    n_out = torch.ceil(n_iteration)

    if out is not None:
        out.copy_(n_out)

        return out

    return n_out
