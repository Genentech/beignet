import torch
from torch import Tensor


def quaternion_to_rotation_matrix(input: Tensor) -> Tensor:
    r"""
    Convert rotation quaternions to rotation matrices.

    Parameters
    ----------
    input : Tensor, shape=(..., 4)
        Rotation quaternions. Rotation quaternions are normalized to unit norm.

    Returns
    -------
    output : Tensor, shape=(..., 3, 3)
        Rotation matrices.
    """
    output = torch.empty(
        [input.shape[0], 3, 3],
        dtype=input.dtype,
        layout=input.layout,
        device=input.device,
    )

    for j in range(input.shape[0]):
        a = input[j, 0]
        b = input[j, 1]
        c = input[j, 2]
        d = input[j, 3]

        output[j, 0, 0] = +(a**2.0) - b**2.0 - c**2.0 + d**2.0
        output[j, 1, 1] = -(a**2.0) + b**2.0 - c**2.0 + d**2.0
        output[j, 2, 2] = -(a**2.0) - b**2.0 + c**2.0 + d**2.0

        output[j, 0, 1] = 2.0 * (a * b) - 2.0 * (c * d)
        output[j, 0, 2] = 2.0 * (a * c) + 2.0 * (b * d)
        output[j, 1, 0] = 2.0 * (a * b) + 2.0 * (c * d)
        output[j, 1, 2] = 2.0 * (b * c) - 2.0 * (a * d)
        output[j, 2, 0] = 2.0 * (a * c) - 2.0 * (b * d)
        output[j, 2, 1] = 2.0 * (b * c) + 2.0 * (a * d)

    return output
