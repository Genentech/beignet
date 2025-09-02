import torch
from torch import Tensor


def compose_rotation_matrix(
    input: Tensor,
    other: Tensor,
) -> Tensor:
    r"""
    Compose rotation matrices.

    This function performs matrix multiplication to compose two rotation matrices.
    For rotation matrices R1 and R2, the composition R1 @ R2 represents applying
    R2 first, then R1.

    Parameters
    ----------
    input : Tensor, shape=(..., 3, 3)
        Rotation matrices.

    other : Tensor, shape=(..., 3, 3)
        Rotation matrices.

    Returns
    -------
    output : Tensor, shape=(..., 3, 3)
        Composed rotation matrices.

    Notes
    -----
    This implementation uses direct matrix multiplication instead of converting
    to quaternions and back, which is more efficient and compatible with torch.compile.
    The mathematical result is identical to the quaternion-based approach.
    """
    return torch.matmul(input, other)
