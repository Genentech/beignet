import torch
from torch import Tensor


def rotation_matrix_to_quaternion(
    input: Tensor,
    canonical: bool | None = False,
) -> Tensor:
    r"""
    Convert rotation matrices to rotation quaternions.

    Parameters
    ----------
    input : Tensor, shape=(..., 3, 3)
        Rotation matrices.

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
    indexes = torch.empty(
        [4],
        dtype=input.dtype,
        layout=input.layout,
        device=input.device,
    )

    output = torch.empty(
        [input.shape[0], 4],
        dtype=input.dtype,
        layout=input.layout,
        device=input.device,
    )

    for j in range(input.shape[0]):
        indexes[0] = input[j, 0, 0]
        indexes[1] = input[j, 1, 1]
        indexes[2] = input[j, 2, 2]
        indexes[3] = input[j, 0, 0] + input[j, 1, 1] + input[j, 2, 2]

        index, maximum = 0, indexes[0]

        for k in range(1, 4):
            if indexes[k] > maximum:
                index, maximum = k, indexes[k]

        if index == 3:
            output[j, 0] = input[j, 2, 1] - input[j, 1, 2]
            output[j, 1] = input[j, 0, 2] - input[j, 2, 0]
            output[j, 2] = input[j, 1, 0] - input[j, 0, 1]
            output[j, 3] = 1.0 + indexes[3]
        else:
            t = index
            u = (t + 1) % 3
            v = (u + 1) % 3

            output[j, t] = 1.0 - indexes[3] + 2.0 * input[j, t, t]
            output[j, u] = input[j, u, t] + input[j, t, u]
            output[j, v] = input[j, v, t] + input[j, t, v]
            output[j, 3] = input[j, v, u] - input[j, u, v]

        a = output[j, 0] ** 2.0
        b = output[j, 1] ** 2.0
        c = output[j, 2] ** 2.0
        d = output[j, 3] ** 2.0

        output[j] = output[j] / torch.sqrt(a + b + c + d)

    if canonical:
        for j in range(output.shape[0]):
            a = output[j, 0]
            b = output[j, 1]
            c = output[j, 2]
            d = output[j, 3]

            if d == 0 and (a == 0 & (b == 0 & c < 0 | b < 0) | a < 0) | d < 0:
                output[j] = -output[j]

    return output
