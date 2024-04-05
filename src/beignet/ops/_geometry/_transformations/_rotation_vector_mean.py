from torch import Tensor

from ._quaternion_mean import quaternion_mean
from ._quaternion_to_rotation_vector import quaternion_to_rotation_vector
from ._rotation_vector_to_quaternion import rotation_vector_to_quaternion


def rotation_vector_mean(
    input: Tensor,
    weight: Tensor | None = None,
    degrees: bool | None = False,
) -> Tensor:
    r"""
    Compose rotation vectors.

    Parameters
    ----------
    input : Tensor, shape=(..., 4)
        Rotation vectors.

    weight : Tensor, shape=(..., 4), optional
        Relative importance of rotation matrices.

    degrees : bool, optional
        If `True`, rotation vector magnitudes are assumed to be in degrees.
        Default, `False`.

    Returns
    -------
    output : Tensor, shape=(..., 4)
        Rotation vectors mean.
    """
    return quaternion_to_rotation_vector(
        quaternion_mean(
            rotation_vector_to_quaternion(
                input,
                degrees,
            ),
            weight,
        ),
        degrees,
    )
