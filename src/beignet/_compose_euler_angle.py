from torch import Tensor

from ._compose_quaternion import compose_quaternion
from ._euler_angle_to_quaternion import euler_angle_to_quaternion
from ._quaternion_to_euler_angle import quaternion_to_euler_angle


def compose_euler_angle(
    input: Tensor,
    other: Tensor,
    axes: str,
    degrees: bool | None = False,
) -> Tensor:
    r"""
    Compose rotation quaternions.

    Parameters
    ----------
    input : Tensor, shape=(..., 3)
        Euler angles.

    other : Tensor, shape=(..., 3)
        Euler angles.

    axes : str
        Axes. One to three characters belonging to the set :math:`\{X, Y, Z\}`
        for intrinsic rotations, or :math:`\{x, y, z\}` for extrinsic
        rotations. Extrinsic and intrinsic rotations cannot be mixed.

    degrees : bool, optional
        If `True`, Euler angles are assumed to be in degrees. Default, `False`.

    Returns
    -------
    output : Tensor, shape=(..., 3)
        Composed Euler angles.
    """
    return quaternion_to_euler_angle(
        compose_quaternion(
            euler_angle_to_quaternion(
                input,
                axes,
                degrees,
            ),
            euler_angle_to_quaternion(
                other,
                axes,
                degrees,
            ),
        ),
        axes,
        degrees,
    )
