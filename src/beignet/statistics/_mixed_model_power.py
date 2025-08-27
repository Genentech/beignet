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

    input = torch.atleast_1d(torch.as_tensor(input))

    n_subjects = torch.atleast_1d(torch.as_tensor(n_subjects))

    n_observations_per_subject = torch.atleast_1d(
        torch.as_tensor(n_observations_per_subject),
    )
    icc = torch.atleast_1d(torch.as_tensor(icc))

    dtypes = [
        input.dtype,
        n_subjects.dtype,
        n_observations_per_subject.dtype,
        icc.dtype,
    ]
    if any(dt == torch.float64 for dt in dtypes):
        dtype = torch.float64
    else:
        dtype = torch.float32

    input = input.to(dtype)

    n_subjects = n_subjects.to(dtype)

    n_observations_per_subject = n_observations_per_subject.to(dtype)

    icc = icc.to(dtype)

    input = torch.clamp(input, min=0.0)

    n_subjects = torch.clamp(n_subjects, min=3.0)

    n_observations_per_subject = torch.clamp(n_observations_per_subject, min=1.0)

    icc = torch.clamp(icc, min=0.0, max=0.99)

    design_effect = 1.0 + (n_observations_per_subject - 1.0) * icc

    total_observations = n_subjects * n_observations_per_subject

    effective_n = total_observations / design_effect

    noncentrality = input * torch.sqrt(effective_n / 4.0)

    df_approximate = n_subjects - 2.0

    df_approximate = torch.clamp(df_approximate, min=1.0)

    sqrt2 = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt2

    t_critical = z_alpha * torch.sqrt(1.0 + 1.0 / (2.0 * df_approximate))

    z_score = t_critical - noncentrality

    power = 0.5 * (1 - torch.erf(z_score / sqrt2))

    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)

        return out

    return power
