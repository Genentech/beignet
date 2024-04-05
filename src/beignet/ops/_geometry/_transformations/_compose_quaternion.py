import torch
from torch import Tensor


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
    output = torch.empty(
        [max(input.shape[0], other.shape[0]), 4],
        dtype=input.dtype,
        layout=input.layout,
        device=input.device,
    )

    for j in range(max(input.shape[0], other.shape[0])):
        a = input[j, 0]
        b = input[j, 1]
        c = input[j, 2]
        d = input[j, 3]

        p = other[j, 0]
        q = other[j, 1]
        r = other[j, 2]
        s = other[j, 3]

        t = output[j, 0]
        u = output[j, 1]
        v = output[j, 2]
        w = output[j, 3]

        output[j, 0] = d * p + s * a + b * r - c * q
        output[j, 1] = d * q + s * b + c * p - a * r
        output[j, 2] = d * r + s * c + a * q - b * p
        output[j, 3] = d * s - a * p - b * q - c * r

        x = torch.sqrt(t**2.0 + u**2.0 + v**2.0 + w**2.0)

        if x == 0.0:
            output[j] = torch.nan

        output[j] = output[j] / x

        if canonical:
            if w == 0 and (t == 0 and (u == 0 and v < 0 or u < 0) or t < 0) or w < 0:
                output[j] = -output[j]

    return output
