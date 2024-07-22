import torch
from torch import Tensor


def rotation_vector_to_quaternion(
    input: Tensor,
    degrees: bool | None = False,
    canonical: bool | None = False,
) -> Tensor:
    r"""
    Convert rotation vector to rotation quaternion.

    Parameters
    ----------
    input : Tensor, shape=(..., 3)
        Rotation vector.

    degrees : bool, optional
        If `True`, rotation vector magnitudes are assumed to be in degrees.
        Default, `False`.

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
        Rotation quaternion.
    """
    if degrees:
        input = torch.deg2rad(input)

    output = torch.empty(
        [input.shape[0], 4],
        dtype=input.dtype,
        layout=input.layout,
        device=input.device,
    )

    for j in range(input.shape[0]):
        t = input[j, 0] ** 2.0
        u = input[j, 1] ** 2.0
        v = input[j, 2] ** 2.0

        y = torch.sqrt(t + u + v)

        if y < 0.001:
            scale = 0.5 - y**2.0 / 48.0 + y**2.0 * y**2.0 / 3840.0
        else:
            scale = torch.sin(y / 2) / y

        output[j, :-1] = input[j] * scale

        output[j, 3] = torch.cos(y / 2)

    if canonical:
        for j in range(output.shape[0]):
            a = output[j, 0]
            b = output[j, 1]
            c = output[j, 2]
            d = output[j, 3]

            if d == 0 and (a == 0 & (b == 0 & c < 0 | b < 0) | a < 0) | d < 0:
                output[j] = -output[j]

    return output
