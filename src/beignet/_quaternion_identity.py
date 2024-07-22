import torch
from torch import Tensor


def quaternion_identity(
    size: int,
    canonical: bool | None = False,
    *,
    out: Tensor | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = torch.strided,
    device: torch.device | None = None,
    requires_grad: bool | None = False,
) -> Tensor:
    r"""
    Identity rotation quaternions.

    Parameters
    ----------
    size : int
        Output size.

    canonical : bool, optional
        Whether to map the redundant double cover of rotation space to a unique
        canonical single cover. If `True`, then the rotation quaternion is
        chosen from :math:`{q, -q}` such that the :math:`w` term is positive.
        If the :math:`w` term is :math:`0`, then the rotation quaternion is
        chosen such that the first non-zero term of the :math:`x`, :math:`y`,
        and :math:`z` terms is positive.

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
    identity_quaternions : Tensor, shape (size, 4)
        Identity rotation quaternions.
    """
    rotation = torch.zeros(
        [size, 4],
        out=out,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )

    rotation[:, 3] = 1.0

    return rotation
