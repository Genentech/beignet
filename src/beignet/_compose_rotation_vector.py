from torch import Tensor

from ._compose_quaternion import compose_quaternion
from ._quaternion_to_rotation_vector import quaternion_to_rotation_vector
from ._rotation_vector_to_quaternion import rotation_vector_to_quaternion


def compose_rotation_vector(
    input: Tensor,
    other: Tensor,
    degrees: bool | None = False,
) -> Tensor:
    r"""
    Compose rotation vectors.

    Parameters
    ----------
    input : Tensor, shape=(..., 4)
        Rotation vectors.

    other : Tensor, shape=(..., 4)
        Rotation vectors.

    degrees : bool, optional
        If `True`, rotation vector magnitudes are assumed to be in degrees.
        Default, `False`.

    Returns
    -------
    output : Tensor, shape=(..., 4)
        Composed rotation vectors.
    """
    return quaternion_to_rotation_vector(
        compose_quaternion(
            rotation_vector_to_quaternion(
                input,
                degrees,
            ),
            rotation_vector_to_quaternion(
                other,
                degrees,
            ),
        ),
        degrees,
    )
