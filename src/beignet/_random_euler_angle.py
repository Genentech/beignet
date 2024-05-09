import torch
from torch import Generator, Tensor

from ._quaternion_to_euler_angle import (
    quaternion_to_euler_angle,
)
from ._random_quaternion import random_quaternion


def random_euler_angle(
    size: int,
    axes: str,
    degrees: bool | None = False,
    *,
    generator: Generator | None = None,
    out: Tensor | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = torch.strided,
    device: torch.device | None = None,
    requires_grad: bool | None = False,
    pin_memory: bool | None = False,
) -> Tensor:
    r"""
    Generate random Euler angles.

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

    generator : torch.Generator, optional
        Psuedo-random number generator. Default, `None`.

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

    pin_memory : bool, optional
        If `True`, returned tensor is allocated in pinned memory. Default,
        `False`.

    Returns
    -------
    random_euler_angles : Tensor, shape (..., 3)
        Random Euler angles.

        The returned Euler angles are in the range:

            *   First angle: :math:`(-180, 180]` degrees (inclusive)
            *   Second angle:
                    *   :math:`[-90, 90]` degrees if all axes are different
                        (e.g., :math:`xyz`)
                    *   :math:`[0, 180]` degrees if first and third axes are
                        the same (e.g., :math:`zxz`)
            *   Third angle: :math:`[-180, 180]` degrees (inclusive)
    """
    return quaternion_to_euler_angle(
        random_quaternion(
            size,
            generator=generator,
            out=out,
            dtype=dtype,
            layout=layout,
            device=device,
            requires_grad=requires_grad,
            pin_memory=pin_memory,
        ),
        axes,
        degrees,
    )
