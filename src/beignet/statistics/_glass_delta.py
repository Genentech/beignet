import torch
from torch import Tensor


def glass_delta(
    x: Tensor,
    y: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
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

    mean_x = torch.mean(x, dim=-1)
    mean_y = torch.mean(y, dim=-1)

    var_y = torch.var(y, dim=-1, correction=1)

    std_y = torch.sqrt(torch.clamp(var_y, min=torch.finfo(dtype).eps))

    delta = (mean_x - mean_y) / std_y

    if out is not None:
        out.copy_(delta)
        return out
    return delta
