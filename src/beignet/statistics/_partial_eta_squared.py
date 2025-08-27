import torch
from torch import Tensor


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

    dtype = torch.promote_types(ss_effect.dtype, ss_error.dtype)

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
