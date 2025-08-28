import functools
import math

import torch
from torch import Tensor

import beignet.distributions


def repeated_measures_analysis_of_variance_power(
    input: Tensor,
    n_subjects: Tensor,
    n_timepoints: Tensor,
    epsilon: Tensor = 1.0,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    input : Tensor
        Input tensor.
    n_subjects : Tensor
        N Subjects parameter.
    n_timepoints : Tensor
        Time parameter.
    epsilon : Tensor, default 1.0
        Epsilon parameter.
    alpha : float, default 0.05
        Type I error rate.
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        Statistical power.
    """

    input = torch.atleast_1d(input)

    n_subjects = torch.atleast_1d(n_subjects)

    n_timepoints = torch.atleast_1d(n_timepoints)

    epsilon = torch.atleast_1d(epsilon)

    dtype = functools.reduce(
        torch.promote_types,
        [input.dtype, n_subjects.dtype, n_timepoints.dtype, epsilon.dtype],
    )

    input = input.to(dtype)

    n_subjects = n_subjects.to(dtype)

    n_timepoints = n_timepoints.to(dtype)

    epsilon = epsilon.to(dtype)

    input = torch.clamp(input, min=0.0)

    n_subjects = torch.clamp(n_subjects, min=3.0)

    n_timepoints = torch.clamp(n_timepoints, min=2.0)

    epsilon = torch.maximum(epsilon, 1.0 / (n_timepoints - 1.0))

    epsilon = torch.clamp(epsilon, max=1.0)

    variance_f = (
        2
        * ((n_timepoints - 1.0) * epsilon + 2 * n_subjects * (input**2) * n_timepoints)
        / (((n_timepoints - 1.0) * epsilon) ** 2)
        * (
            (
                torch.clamp(
                    (n_subjects - 1.0) * (n_timepoints - 1.0) * epsilon,
                    min=1.0,
                )
                + 2.0
            )
            / torch.clamp(
                torch.clamp(
                    (n_subjects - 1.0) * (n_timepoints - 1.0) * epsilon,
                    min=1.0,
                ),
                min=1.0,
            )
        )
    )

    power = torch.clamp(
        0.5
        * (
            1
            - torch.erf(
                (
                    beignet.distributions.FisherSnedecor(
                        (n_timepoints - 1.0) * epsilon,
                        (n_subjects - 1.0) * (n_timepoints - 1.0),
                    ).icdf(torch.tensor(1 - alpha, dtype=dtype))
                    - (
                        (n_timepoints - 1.0) * epsilon
                        + n_subjects * (input**2) * n_timepoints
                    )
                    / ((n_timepoints - 1.0) * epsilon)
                )
                / torch.sqrt(torch.clamp(variance_f, min=torch.finfo(dtype).eps))
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
