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

    input = input / torch.linalg.vector_norm(input, ord=2, dim=-1, keepdim=True)

    a, b, c, d = torch.unbind(input, dim=-1)

    return torch.stack(
        [
            torch.stack(
                [
                    a.pow(2) - b.pow(2) - c.pow(2) + d.pow(2),
                    2 * (a * b) - 2 * (c * d),
                    2 * (a * c) + 2 * (b * d),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2 * (a * b) + 2 * (c * d),
                    -a.pow(2) + b.pow(2) - c.pow(2) + d.pow(2),
                    2 * (b * c) - 2 * (a * d),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2 * (a * c) - 2 * (b * d),
                    2 * (b * c) + 2 * (a * d),
                    -a.pow(2) - b.pow(2) + c.pow(2) + d.pow(2),
                ],
                dim=-1,
            ),
        ],
        dim=-2,
    )
