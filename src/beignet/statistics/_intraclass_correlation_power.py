import math

import torch
from torch import Tensor


def intraclass_correlation_power(
    icc: Tensor,
    n_subjects: Tensor,
    n_raters: Tensor,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    icc : Tensor
        Icc parameter.
    n_subjects : Tensor
        N Subjects parameter.
    n_raters : Tensor
        N Raters parameter.
    alpha : float, default 0.05
        Type I error rate.
    alternative : str, default 'two-sided'
        Alternative hypothesis ("two-sided", "greater", "less").
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        Statistical power.
    """

    icc = torch.atleast_1d(icc)

    n_subjects = torch.atleast_1d(n_subjects)

    n_raters = torch.atleast_1d(n_raters)

    dtype = torch.promote_types(icc.dtype, n_subjects.dtype)
    dtype = torch.promote_types(dtype, n_raters.dtype)
    icc = icc.to(dtype)

    n_subjects = n_subjects.to(dtype)

    n_raters = n_raters.to(dtype)

    icc = torch.clamp(icc, min=0.0, max=0.99)

    n_subjects = torch.clamp(n_subjects, min=3.0)

    n_raters = torch.clamp(n_raters, min=2.0)

    df_between = n_subjects - 1

    f_expected = (1 + (n_raters - 1) * icc) / (1 - icc)

    sqrt2 = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2

    f_critical = 1.0 + z_alpha * torch.sqrt(2.0 / df_between)

    mean_f = f_expected

    variance_f = 2 * f_expected * f_expected / df_between

    standard_deviation_f = torch.sqrt(
        torch.clamp(variance_f, min=torch.finfo(dtype).eps),
    )

    z_score = (f_critical - mean_f) / standard_deviation_f

    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError(
            f"alternative must be 'two-sided', 'greater', or 'less', got {alternative}",
        )

    if alternative == "two-sided":
        power = 0.5 * (1 - torch.erf(torch.abs(z_score) / sqrt2))
    elif alternative == "greater":
        power = 0.5 * (1 - torch.erf(z_score / sqrt2))
    else:
        power = 0.5 * (1 + torch.erf(z_score / sqrt2))

    if out is not None:
        out.copy_(power)

        return out

    return power
