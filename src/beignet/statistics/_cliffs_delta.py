import torch
from torch import Tensor


def cliffs_delta(
    x: Tensor,
    y: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor:
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

    x_expanded = x.unsqueeze(-1)
    y_expanded = y.unsqueeze(-2)

    steepness_parameter = 100.0

    diff = x_expanded - y_expanded

    n_xy = torch.sum(torch.sigmoid(steepness_parameter * diff), dim=(-2, -1))

    n_yx = torch.sum(torch.sigmoid(steepness_parameter * (-diff)), dim=(-2, -1))

    n_x = torch.tensor(x.shape[-1], dtype=dtype)
    n_y = torch.tensor(y.shape[-1], dtype=dtype)

    total_comparisons = n_x * n_y

    delta = (n_xy - n_yx) / total_comparisons

    if out is not None:
        out.copy_(delta)
        return out
    return delta
