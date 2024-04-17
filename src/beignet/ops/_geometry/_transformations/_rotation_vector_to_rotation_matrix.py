from torch import Tensor

from ._quaternion_to_rotation_matrix import (
    quaternion_to_rotation_matrix,
)
from ._rotation_vector_to_quaternion import (
    rotation_vector_to_quaternion,
)


def rotation_vector_to_rotation_matrix(
    input: Tensor,
    degrees: bool | None = False,
) -> Tensor:
    r"""
    Convert rotation vectors to rotation matrices.

    Parameters
    ----------
    input : Tensor, shape=(..., 3)
        Rotation vectors.

    degrees : bool, optional
        If `True`, rotation vector magnitudes are assumed to be in degrees.
        Default, `False`.

    Returns
    -------
    output : Tensor, shape=(..., 3, 3)
        Rotation matrices.
    """
    return quaternion_to_rotation_matrix(
        rotation_vector_to_quaternion(
            input,
            degrees,
        ),
    )
