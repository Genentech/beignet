import math

import torch
from torch import Tensor

from ._mixed_model_power import mixed_model_power


def mixed_model_sample_size(
    input: Tensor,
    n_observations_per_subject: Tensor,
    icc: Tensor,
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
    n_observations_per_subject : Tensor
        N Observations Per Subject parameter.
    icc : Tensor
        Icc parameter.
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

    n_observations_per_subject = torch.atleast_1d(
        torch.as_tensor(n_observations_per_subject),
    )
    icc = torch.atleast_1d(torch.as_tensor(icc))

    dtypes = [input.dtype, n_observations_per_subject.dtype, icc.dtype]
    if any(dt == torch.float64 for dt in dtypes):
        dtype = torch.float64
    else:
        dtype = torch.float32

    input = input.to(dtype)

    n_observations_per_subject = n_observations_per_subject.to(dtype)

    icc = icc.to(dtype)

    input = torch.clamp(input, min=torch.finfo(dtype).eps)

    n_observations_per_subject = torch.clamp(n_observations_per_subject, min=1.0)

    icc = torch.clamp(icc, min=0.0, max=0.99)

    n_iteration = torch.clamp(
        4
        * (
            (
                torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * math.sqrt(2.0)
                + torch.erfinv(torch.tensor(power, dtype=dtype)) * math.sqrt(2.0)
            )
            / input
        )
        ** 2
        * (1.0 + (n_observations_per_subject - 1.0) * icc)
        / n_observations_per_subject,
        min=10.0,
    )
    for _ in range(15):
        n_iteration = torch.clamp(
            n_iteration
            * (
                1.0
                + 1.3
                * torch.clamp(
                    power
                    - mixed_model_power(
                        input,
                        n_iteration,
                        n_observations_per_subject,
                        icc,
                        alpha=alpha,
                    ),
                    min=-0.4,
                    max=0.4,
                )
            ),
            min=10.0,
            max=1e5,
        )

    n_out = torch.ceil(n_iteration)

    if out is not None:
        out.copy_(n_out)

        return out

    return n_out
