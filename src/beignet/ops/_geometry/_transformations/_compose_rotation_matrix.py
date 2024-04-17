from torch import Tensor

from ._compose_quaternion import compose_quaternion
from ._quaternion_to_rotation_matrix import quaternion_to_rotation_matrix
from ._rotation_matrix_to_quaternion import rotation_matrix_to_quaternion


def compose_rotation_matrix(
    input: Tensor,
    other: Tensor,
) -> Tensor:
    r"""
    Compose rotation matrices.

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
    """
    return quaternion_to_rotation_matrix(
        compose_quaternion(
            rotation_matrix_to_quaternion(input),
            rotation_matrix_to_quaternion(other),
        ),
    )
