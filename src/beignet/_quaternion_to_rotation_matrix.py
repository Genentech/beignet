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

    a, b, c, d = torch.unbind(input, dim=-1)

    return torch.stack(
        [
            torch.square(a) - torch.square(b) - torch.square(c) + torch.square(d),
            2 * (a * b) - 2 * (c * d),
            2 * (a * c) + 2 * (b * d),
            2 * (a * b) + 2 * (c * d),
            -torch.square(a) + torch.square(b) - torch.square(c) + torch.square(d),
            2 * (b * c) - 2 * (a * d),
            2 * (a * c) - 2 * (b * d),
            2 * (b * c) + 2 * (a * d),
            -torch.square(a) - torch.square(b) + torch.square(c) + torch.square(d),
        ],
        dim=-1,
    ).view(input.shape[:-1] + (3, 3))
