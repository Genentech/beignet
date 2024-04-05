from torch import Tensor

from ._euler_angle_to_quaternion import (
    euler_angle_to_quaternion,
)
from ._quaternion_magnitude import quaternion_magnitude


def euler_angle_magnitude(
    input: Tensor,
    axes: str,
    degrees: bool | None = False,
) -> Tensor:
    r"""
    Euler angle magnitudes.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        Euler angles.

    axes : str
        Axes. 1-3 characters belonging to the set {‘X’, ‘Y’, ‘Z’} for intrinsic
        rotations, or {‘x’, ‘y’, ‘z’} for extrinsic rotations. Extrinsic and
        intrinsic rotations cannot be mixed.

    degrees : bool, optional
        If `True`, Euler angles are assumed to be in degrees. Default, `False`.

    Returns
    -------
    euler_angle_magnitudes: Tensor, shape (...)
        Angles in radians. Magnitudes will be in the range :math:`[0, \pi]`.
    """
    return quaternion_magnitude(
        euler_angle_to_quaternion(
            input,
            axes,
            degrees,
        ),
    )
