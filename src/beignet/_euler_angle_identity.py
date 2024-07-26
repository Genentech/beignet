import torch
from torch import Tensor

from ._quaternion_identity import quaternion_identity
from ._quaternion_to_euler_angle import (
    quaternion_to_euler_angle,
)


def euler_angle_identity(
    size: int,
    axes: str,
    degrees: bool | None = False,
    *,
    out: Tensor | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = torch.strided,
    device: torch.device | None = None,
    requires_grad: bool | None = False,
) -> Tensor:
    r"""
    Identity Euler angles.

    Parameters
    ----------
    size : int
        Output size.

    axes : str
        Axes. 1-3 characters belonging to the set {‘X’, ‘Y’, ‘Z’} for intrinsic
        rotations, or {‘x’, ‘y’, ‘z’} for extrinsic rotations. Extrinsic and
        intrinsic rotations cannot be mixed.

    degrees : bool, optional
        If `True`, Euler angles are assumed to be in degrees. Default, `False`.

    out : Tensor, optional
        Output tensor. Default, `None`.

    dtype : torch.dtype, optional
        Type of the returned tensor. Default, global default.

    layout : torch.layout, optional
        Layout of the returned tensor. Default, `torch.strided`.

    device : torch.device, optional
        Device of the returned tensor. Default, current device for the default
        tensor type.

    requires_grad : bool, optional
        Whether autograd records operations on the returned tensor. Default,
        `False`.

    Returns
    -------
    identity_euler_angles : Tensor, shape (size, 3)
        Identity Euler angles.
    """
    return quaternion_to_euler_angle(
        quaternion_identity(
            size,
            out=out,
            dtype=dtype,
            layout=layout,
            device=device,
            requires_grad=requires_grad,
        ),
        axes,
        degrees,
    )
