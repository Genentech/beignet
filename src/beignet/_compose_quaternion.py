import torch
from torch import Tensor

from ._canonicalize_quaternion import canonicalize_quaternion


def compose_quaternion(
    input: Tensor,
    other: Tensor,
    canonical: bool = False,
) -> Tensor:
    r"""
    Compose rotation quaternions.

    Parameters
    ----------
    input : Tensor, shape=(..., 4)
        Rotation quaternions. Rotation quaternions are normalized to unit norm.

    other : Tensor, shape=(..., 4)
        Rotation quaternions. Rotation quaternions are normalized to unit norm.

    canonical : bool, optional
        Whether to map the redundant double cover of rotation space to a unique
        canonical single cover. If `True`, then the rotation quaternion is
        chosen from :math:`{q, -q}` such that the :math:`w` term is positive.
        If the :math:`w` term is :math:`0`, then the rotation quaternion is
        chosen such that the first non-zero term of the :math:`x`, :math:`y`,
        and :math:`z` terms is positive.

    Returns
    -------
    output : Tensor, shape=(..., 4)
        Composed rotation quaternions.
    """

    a, b, c, d = torch.unbind(input, dim=-1)
    p, q, r, s = torch.unbind(other, dim=-1)

    t = d * p + s * a + b * r - c * q
    u = d * q + s * b + c * p - a * r
    v = d * r + s * c + a * q - b * p
    w = d * s - a * p - b * q - c * r

    output = torch.stack([t, u, v, w], dim=-1)

    x = torch.sqrt(torch.sum(torch.square(output), dim=-1, keepdim=True))

    output = torch.where(x == 0.0, torch.nan, output / x)

    if canonical:
        return canonicalize_quaternion(output)

    return output
