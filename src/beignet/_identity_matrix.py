import torch
from torch import Tensor


def identity_matrix(
    d: int, size: tuple[int, ...] = (), dtype=None, device=None
) -> Tensor:
    """Return identity matrix.

    Parameters
    ----------
    d: int
        Dimension of matrix

    size: tuple[int, ...]
        Batch dimensions

    dtype = None
        dtype

    device = None
        device

    Returns
    -------
    Tensor
        Identity matrix with leading batch dimensions
        Shape (*size, d, d)
    """
    return (
        torch.eye(d, dtype=dtype, device=device)
        .view(*((1,) * len(size)), d, d)
        .repeat(*size, 1, 1)
    )
