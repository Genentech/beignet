import math

import torch
from torch import Tensor

from beignet.distributions import Chi2, NonCentralChi2


def friedman_test_power(
    input: Tensor,
    subjects: Tensor,
    treatments: Tensor,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    input : Tensor
        Input tensor.
    subjects : Tensor
        N Subjects parameter.
    treatments : Tensor
        N Treatments parameter.
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

    subjects = torch.atleast_1d(subjects)

    treatments = torch.atleast_1d(treatments)

    dtype = torch.promote_types(input.dtype, subjects.dtype)
    dtype = torch.promote_types(dtype, treatments.dtype)

    subjects = subjects.to(dtype)

    treatments = treatments.to(dtype)

    input = torch.clamp(input, min=0.0)
    subjects = torch.clamp(subjects, min=3.0)
    treatments = torch.clamp(treatments, min=3.0)

    distribution = Chi2(treatments - 1)
    critical_value = distribution.icdf(torch.tensor(1 - alpha, dtype=dtype))

    distribution = NonCentralChi2(
        treatments - 1,
        12 * subjects * input / (treatments * (treatments + 1)),
    )

    output = torch.clamp(distribution.variance, min=torch.finfo(dtype).eps)
    output = torch.sqrt(output)
    output = (critical_value - distribution.mean) / output
    output = output / math.sqrt(2.0)

    output = torch.erf(output)
    output = 1.0 - output
    output = 0.5 * output

    if out is not None:
        out.copy_(output)

        return out

    return output
