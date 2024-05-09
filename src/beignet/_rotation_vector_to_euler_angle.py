from torch import Tensor

from ._quaternion_to_euler_angle import (
    quaternion_to_euler_angle,
)
from ._rotation_vector_to_quaternion import (
    rotation_vector_to_quaternion,
)


def rotation_vector_to_euler_angle(
    input: Tensor,
    axes: str,
    degrees: bool = False,
) -> Tensor:
    r"""
    Convert rotation vectors to Euler angles.

    Parameters
    ----------
    input : Tensor, shape=(..., 3)
        Rotation vectors.

    degrees : bool, optional
        If `True`, rotation vector magnitudes are assumed to be in degrees.
        Default, `False`.

    Returns
    -------
    output : Tensor, shape=(..., 3)
        Euler angles. The returned Euler angles are in the range:

            * First angle: :math:`(-180, 180]` degrees (inclusive)
            * Second angle:
                * :math:`[-90, 90]` degrees if all axes are different
                  (e.g., :math:`xyz`)
                * :math:`[0, 180]` degrees if first and third axes are the same
                  (e.g., :math:`zxz`)
            * Third angle: :math:`[-180, 180]` degrees (inclusive)
    """
    return quaternion_to_euler_angle(
        rotation_vector_to_quaternion(
            input,
            degrees,
        ),
        axes,
        degrees,
    )
