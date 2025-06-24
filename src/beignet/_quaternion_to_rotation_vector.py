import torch
from torch import Tensor

from ._canonicalize_quaternion import canonicalize_quaternion


def quaternion_to_rotation_vector(
    input: Tensor,
    degrees: bool | None = False,
) -> Tensor:
    r"""
    Convert rotation quaternions to rotation vectors.

    Parameters
    ----------
    input : Tensor, shape=(..., 4)
        Rotation quaternions. Rotation quaternions are normalized to unit norm.

    degrees : bool, optional

    Returns
    -------
    output : Tensor, shape=(..., 3)
        Rotation vectors.
    """

    input = canonicalize_quaternion(input)

    a, b, c, d = torch.unbind(input, dim=-1)

    y = 2 * torch.atan2(
        torch.sqrt(torch.square(a) + torch.square(b) + torch.square(c)), d
    )
    y2 = torch.square(y)

    scale = torch.where(
        y < 0.001, 2.0 + y2 / 12 + 7 * y2 * y2 / 2880, y / torch.sin(y / 2.0)
    )

    if degrees:
        scale = torch.rad2deg(scale)

    return scale[..., None] * input[..., :-1]
