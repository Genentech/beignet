from torch import Tensor

from ._euler_angle_to_quaternion import (
    euler_angle_to_quaternion,
)
from ._invert_quaternion import invert_quaternion
from ._quaternion_to_euler_angle import (
    quaternion_to_euler_angle,
)


def invert_euler_angle(
    input: Tensor,
    axes: str,
    degrees: bool | None = False,
) -> Tensor:
    r"""
    Invert Euler angles.

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
    inverted_euler_angles : Tensor, shape (..., 3)
        Inverted Euler angles.
    """
    return quaternion_to_euler_angle(
        invert_quaternion(
            euler_angle_to_quaternion(
                input,
                axes,
                degrees,
            ),
        ),
        axes,
        degrees,
    )
