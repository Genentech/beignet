import torch
from torch import Tensor

from ._quaternion_identity import quaternion_identity
from ._quaternion_to_rotation_vector import (
    quaternion_to_rotation_vector,
)


def rotation_vector_identity(
    size: int,
    degrees: bool = False,
    *,
    out: Tensor | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = torch.strided,
    device: torch.device | None = None,
    requires_grad: bool | None = False,
) -> Tensor:
    r"""
    Identity rotation vectors.

    Parameters
    ----------
    size : int
        Output size.

    degrees : bool
        If `True`, rotation vector magnitudes are assumed to be in degrees.
        Default, `False`.

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
    identity_rotation_vectors : Tensor, shape (size, 3)
        Identity rotation vectors.
    """
    return quaternion_to_rotation_vector(
        quaternion_identity(
            size,
            out=out,
            dtype=dtype,
            layout=layout,
            device=device,
            requires_grad=requires_grad,
        ),
        degrees,
    )
