import torch
from torch import Tensor


def eta_squared(
    ss_between: Tensor,
    ss_total: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    ss_between : Tensor
        Ss Between parameter.
    ss_total : Tensor
        Ss Total parameter.
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        Computed statistic.
    """

    ss_between = torch.atleast_1d(torch.as_tensor(ss_between))

    ss_total = torch.atleast_1d(torch.as_tensor(ss_total))

    if ss_between.dtype.is_floating_point and ss_total.dtype.is_floating_point:
        if ss_between.dtype == torch.float64 or ss_total.dtype == torch.float64:
            dtype = torch.float64
        else:
            dtype = torch.float32
    else:
        dtype = torch.float32

    ss_between = ss_between.to(dtype)

    ss_total = ss_total.to(dtype)

    ss_between = torch.clamp(ss_between, min=0.0)

    ss_total = torch.clamp(ss_total, min=torch.finfo(dtype).eps)

    ss_between = torch.clamp(ss_between, max=ss_total)

    eta_sq = ss_between / ss_total

    eta_sq = torch.clamp(eta_sq, 0.0, 1.0)

    if out is not None:
        out.copy_(eta_sq)
        return out
    return eta_sq


def partial_eta_squared(
    ss_effect: Tensor,
    ss_error: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    ss_between : Tensor
        Ss Between parameter.
    ss_total : Tensor
        Ss Total parameter.
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        Computed statistic.
    """

    ss_effect = torch.atleast_1d(torch.as_tensor(ss_effect))

    ss_error = torch.atleast_1d(torch.as_tensor(ss_error))

    dtype = (
        torch.float64
        if (ss_effect.dtype == torch.float64 or ss_error.dtype == torch.float64)
        else torch.float32
    )
    ss_effect = ss_effect.to(dtype)

    ss_error = ss_error.to(dtype)

    ss_effect = torch.clamp(ss_effect, min=0.0)

    ss_error = torch.clamp(ss_error, min=torch.finfo(dtype).eps)

    partial_eta_sq = ss_effect / (ss_effect + ss_error)

    partial_eta_sq = torch.clamp(partial_eta_sq, 0.0, 1.0)

    if out is not None:
        out.copy_(partial_eta_sq)
        return out
    return partial_eta_sq
