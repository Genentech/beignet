import functools
import math

import torch
from torch import Tensor


def mixed_model_power(
    input: Tensor,
    n_subjects: Tensor,
    n_observations_per_subject: Tensor,
    icc: Tensor,
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
    n_observations_per_subject : Tensor
        N Observations Per Subject parameter.
    icc : Tensor
        Icc parameter.
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

    n_observations_per_subject = torch.atleast_1d(
        torch.as_tensor(n_observations_per_subject),
    )
    icc = torch.atleast_1d(icc)

    dtype = functools.reduce(
        torch.promote_types,
        [input.dtype, n_subjects.dtype, n_observations_per_subject.dtype, icc.dtype],
    )

    input = input.to(dtype)

    n_subjects = n_subjects.to(dtype)

    n_observations_per_subject = n_observations_per_subject.to(dtype)

    icc = icc.to(dtype)

    input = torch.clamp(input, min=0.0)

    n_subjects = torch.clamp(n_subjects, min=3.0)

    n_observations_per_subject = torch.clamp(n_observations_per_subject, min=1.0)

    icc = torch.clamp(icc, min=0.0, max=0.99)

    power = torch.clamp(
        0.5
        * (
            1
            - torch.erf(
                (
                    torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype))
                    * math.sqrt(2.0)
                    * torch.sqrt(
                        1.0 + 1.0 / (2.0 * torch.clamp(n_subjects - 2.0, min=1.0)),
                    )
                    - input
                    * torch.sqrt(
                        n_subjects
                        * n_observations_per_subject
                        / (1.0 + (n_observations_per_subject - 1.0) * icc)
                        / 4.0,
                    )
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
