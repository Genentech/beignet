import torch
from torch import Tensor


def cliffs_delta(
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
        dtype = torch.float32
        for tensor in (x, y):
            dtype = torch.promote_types(dtype, tensor.dtype)
    else:
        dtype = torch.float32

    x = x.to(dtype)
    y = y.to(dtype)

    x_expanded = x.unsqueeze(-1)
    y_expanded = y.unsqueeze(-2)

    steepness = 100.0

    diff = x_expanded - y_expanded

    n_xy = torch.sum(torch.sigmoid(steepness * diff), dim=(-2, -1))

    n_yx = torch.sum(torch.sigmoid(steepness * (-diff)), dim=(-2, -1))

    n_x = torch.tensor(x.shape[-1], dtype=dtype)
    n_y = torch.tensor(y.shape[-1], dtype=dtype)

    total_comparisons = n_x * n_y

    delta = (n_xy - n_yx) / total_comparisons

    if out is not None:
        out.copy_(delta)
        return out
    return delta
