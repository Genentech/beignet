import torch
from torch import Generator, Tensor


def random_quaternion(
    size: int,
    canonical: bool = False,
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
    Generate random rotation quaternions.

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
    random_quaternions : Tensor, shape (..., 4)
        Random rotation quaternions.
    """
    quaternions = torch.rand(
        [size, 4],
        generator=generator,
        out=out,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
        pin_memory=pin_memory,
    )

    if canonical:
        for index in range(quaternions.size(0)):
            if (
                (quaternions[index][3] < 0)
                or (quaternions[index][3] == 0 and quaternions[index][0] < 0)
                or (
                    quaternions[index][3] == 0
                    and quaternions[index][0] == 0
                    and quaternions[index][1] < 0
                )
                or (
                    quaternions[index][3] == 0
                    and quaternions[index][0] == 0
                    and quaternions[index][1] == 0
                    and quaternions[index][2] < 0
                )
            ):
                quaternions[index][0] *= -1.0
                quaternions[index][1] *= -1.0
                quaternions[index][2] *= -1.0
                quaternions[index][3] *= -1.0

    return quaternions
