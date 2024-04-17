import torch
from torch import Generator, Tensor

from ._quaternion_to_rotation_matrix import (
    quaternion_to_rotation_matrix,
)
from ._random_quaternion import random_quaternion


def random_rotation_matrix(
    size: int,
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
    Generate random rotation matrices.

    Parameters
    ----------
    size : int
        Output size.

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
    random_rotation_matrices : Tensor, shape (..., 3, 3)
        Random rotation matrices.
    """
    return quaternion_to_rotation_matrix(
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
    )
