import torch
from torch import Tensor


def glass_delta(
    x: Tensor,
    y: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    x : Tensor
        X parameter.
    y : Tensor
        Y parameter.
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        Computed statistic.
    """

    x = torch.atleast_1d(torch.as_tensor(x))
    y = torch.atleast_1d(torch.as_tensor(y))

    if x.dtype.is_floating_point and y.dtype.is_floating_point:
        if x.dtype == torch.float64 or y.dtype == torch.float64:
            dtype = torch.float64
        else:
            dtype = torch.float32
    else:
        dtype = torch.float32

    x = x.to(dtype)
    y = y.to(dtype)

    delta = (torch.mean(x, dim=-1) - torch.mean(y, dim=-1)) / torch.sqrt(
        torch.clamp(torch.var(y, dim=-1, correction=1), min=torch.finfo(dtype).eps),
    )

    if out is not None:
        out.copy_(delta)

        return out

    return delta
