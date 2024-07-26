from torch import Tensor

from ._euler_angle_to_quaternion import euler_angle_to_quaternion
from ._quaternion_mean import quaternion_mean
from ._quaternion_to_euler_angle import quaternion_to_euler_angle


def euler_angle_mean(
    input: Tensor,
    weight: Tensor | None = None,
    axes: str | None = None,
    degrees: bool | None = False,
) -> Tensor:
    r"""
    Euler angle mean.

    Parameters
    ----------
    input : Tensor, shape=(..., 3)
        Euler angles.

    weight : Tensor, shape=(..., 4), optional
        Relative importance of rotation quaternions.

    axes : str
        Axes. One to three characters belonging to the set :math:`\{X, Y, Z\}`
        for intrinsic rotations, or :math:`\{x, y, z\}` for extrinsic
        rotations. Extrinsic and intrinsic rotations cannot be mixed.

    degrees : bool, optional
        If `True`, Euler angles are assumed to be in degrees. Default, `False`.

    Returns
    -------
    output : Tensor, shape=(..., 3)
        Euler angle mean.
    """
    return quaternion_to_euler_angle(
        quaternion_mean(
            euler_angle_to_quaternion(
                input,
                axes,
                degrees,
            ),
            weight,
        ),
        axes,
        degrees,
    )
