from torch import Tensor

from ._quaternion_magnitude import quaternion_magnitude
from ._rotation_vector_to_quaternion import (
    rotation_vector_to_quaternion,
)


def rotation_vector_magnitude(
    input: Tensor,
    degrees: bool | None = False,
) -> Tensor:
    r"""
    Rotation vector magnitudes.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        Rotation vectors.

    degrees : bool, optional
        If `True`, magnitudes are assumed to be in degrees. Default, `False`.

    Returns
    -------
    rotation_vector_magnitudes : Tensor, shape (...)
        Angles in radians. Magnitudes will be in the range :math:`[0, \pi]`.
    """
    return quaternion_magnitude(
        rotation_vector_to_quaternion(
            input,
            degrees,
        ),
    )
