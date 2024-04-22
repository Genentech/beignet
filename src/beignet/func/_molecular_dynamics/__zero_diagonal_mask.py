import torch
from torch import Tensor


def _zero_diagonal_mask(x: Tensor) -> Tensor:
    """Sets the diagonal of a matrix to zero."""
    if x.shape[0] != x.shape[1]:
        raise ValueError(
            f"Diagonal mask can only mask square matrices. Found {x.shape[0]}x{x.shape[1]}."
        )

    if len(x.shape) > 3:
        raise ValueError(
            f"Diagonal mask can only mask rank-2 or rank-3 tensors. Found {len(x.shape)}."
        )

    n = x.shape[0]

    x = torch.nan_to_num(x)

    mask = 1.0 - torch.eye(n, device=x.device, dtype=x.dtype)

    if len(x.shape) == 3:
        mask = torch.reshape(mask, [n, n, 1])

    return x * mask
