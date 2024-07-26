from torch import Tensor

from ._quaternion_magnitude import quaternion_magnitude
from ._rotation_matrix_to_quaternion import (
    rotation_matrix_to_quaternion,
)


def rotation_matrix_magnitude(input: Tensor) -> Tensor:
    r"""
    Rotation matrix magnitudes.

    Parameters
    ----------
    input : Tensor, shape (..., 3, 3)
        Rotation matrices.

    Returns
    -------
    rotation_matrix_magnitudes: Tensor, shape (...)
        Angles in radians. Magnitudes will be in the range :math:`[0, \pi]`.
    """
    return quaternion_magnitude(
        rotation_matrix_to_quaternion(
            input,
        ),
    )
