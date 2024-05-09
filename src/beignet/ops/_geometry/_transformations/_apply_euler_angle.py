from torch import Tensor

from ._apply_rotation_matrix import apply_rotation_matrix
from ._euler_angle_to_rotation_matrix import euler_angle_to_rotation_matrix


def apply_euler_angle(
    input: Tensor,
    rotation: Tensor,
    axes: str,
    degrees: bool = False,
    inverse: bool = False,
) -> Tensor:
    r"""
    Rotates vectors in three-dimensional space using Euler angles.

    Note
    ----
    This function interprets the rotation of the original frame to the final
    frame as either a projection, where it maps the components of vectors from
    the final frame to the original frame, or as a physical rotation,
    integrating the vectors into the original frame during the rotation
    process. Consequently, the vector components are maintained in the original
    frame’s perspective both before and after the rotation.

    Parameters
    ----------
    input : Tensor
        Vectors in three-dimensional space with the shape $(\ldots \times 3)$.
        Euler angles and vectors must conform to PyTorch broadcasting rules.

    rotation : Tensor
        Euler angles with the shape $(\ldots \times 3)$, specifying the
        rotation in three-dimensional space.

    axes : str
        Specifies the sequence of axes for the rotations, using one to three
        characters from the set ${X, Y, Z}$ for intrinsic rotations, or
        ${x, y, z}$ for extrinsic rotations. Mixing extrinsic and intrinsic
        rotations raises a `ValueError`.

    degrees : bool, optional
        Indicates whether the Euler angles are provided in degrees. If `False`,
        angles are assumed to be in radians. Default, `False`.

    inverse : bool, optional
        If `True`, applies the inverse rotation using the Euler angles to the
        input vectors. Default, `False`.

    Returns
    -------
    output : Tensor
        A tensor of the same shape as `input`, containing the rotated vectors.
    """
    return apply_rotation_matrix(
        input,
        euler_angle_to_rotation_matrix(
            rotation,
            axes,
            degrees,
        ),
        inverse,
    )
