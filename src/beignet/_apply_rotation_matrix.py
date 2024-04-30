import torch
from torch import Tensor


def apply_rotation_matrix(
    input: Tensor,
    rotation: Tensor,
    inverse: bool | None = False,
) -> Tensor:
    r"""
    Rotates vectors in three-dimensional space using rotation matrices.

    Note
    ----
    This function interprets the rotation of the original frame to the final
    frame as either a projection, where it maps the components of vectors from
    the final frame to the original frame, or as a physical rotation,
    integrating the vectors into the original frame during the rotation
    process. Consequently, the vector components are maintained in the original
    frame’s perspective both before and after the rotation.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        Each vector represents a vector in three-dimensional space. The number
        of rotation matrices and number of vectors must follow standard
        broadcasting rules: either one of them equals unity or they both equal
        each other.

    rotation : Tensor, shape (..., 3, 3)
        Rotation matrices.

    inverse : bool, optional
        If `True` the inverse of the rotation matrices are applied to the input
        vectors. Default, `False`.

    Returns
    -------
    rotated_vectors : Tensor, shape (..., 3)
        Rotated vectors.
    """
    if inverse:
        output = torch.einsum("ikj, ik -> ij", rotation, input)
    else:
        output = torch.einsum("ijk, ik -> ij", rotation, input)

    return output
