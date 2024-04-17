from torch import Tensor

from ._quaternion_to_rotation_vector import (
    quaternion_to_rotation_vector,
)
from ._rotation_matrix_to_quaternion import (
    rotation_matrix_to_quaternion,
)


def rotation_matrix_to_rotation_vector(
    input: Tensor,
    degrees: bool = False,
) -> Tensor:
    r"""
    Convert rotation matrices to rotation vectors.

    Parameters
    ----------
    input : Tensor, shape=(..., 3, 3)
        Rotation matrices.

    degrees : bool
        If `True`, rotation vector magnitudes are assumed to be in degrees.
        Default, `False`.

    Returns
    -------
    output : Tensor, shape=(..., 3)
        Rotation vectors.
    """
    return quaternion_to_rotation_vector(
        rotation_matrix_to_quaternion(
            input,
        ),
        degrees,
    )
