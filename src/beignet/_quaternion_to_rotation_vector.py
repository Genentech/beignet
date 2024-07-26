import torch
from torch import Tensor


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
    output = torch.empty(
        [input.shape[0], 3],
        dtype=input.dtype,
        layout=input.layout,
        device=input.device,
    )

    for j in range(input.shape[0]):
        a = input[j, 0]
        b = input[j, 1]
        c = input[j, 2]
        d = input[j, 3]

        if d == 0 and (a == 0 and (b == 0 and c < 0 or b < 0) or a < 0) or d < 0:
            input[j] = -input[j]

        t = input[j, 0] ** 2.0
        u = input[j, 1] ** 2.0
        v = input[j, 2] ** 2.0
        w = input[j, 3] ** 1.0

        y = 2.0 * torch.atan2(torch.sqrt(t + u + v), w)

        if y < 0.001:
            y = 2.0 + y**2.0 / 12 + 7 * y**2.0 * y**2.0 / 2880
        else:
            y = y / torch.sin(y / 2.0)

        output[j] = input[j, :-1] * y

    if degrees:
        output = torch.rad2deg(output)

    return output
