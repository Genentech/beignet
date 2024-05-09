from torch import Tensor

from ._quaternion_mean import quaternion_mean
from ._quaternion_to_rotation_matrix import quaternion_to_rotation_matrix
from ._rotation_matrix_to_quaternion import rotation_matrix_to_quaternion


def rotation_matrix_mean(
    input: Tensor,
    weight: Tensor | None = None,
) -> Tensor:
    r"""
    Parameters
    ----------
    input : Tensor, shape=(..., 3, 3)
        Rotation matrices.

    weight : Tensor, shape=(..., 4), optional
        Relative importance of rotation matrices.

    Returns
    -------
    output : Tensor, shape=(..., 3, 3)
    """
    return quaternion_to_rotation_matrix(
        quaternion_mean(
            rotation_matrix_to_quaternion(
                input,
            ),
            weight,
        ),
    )
