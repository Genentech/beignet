import torch
from torch import Tensor


def quaternion_mean(
    input: Tensor,
    weight: Tensor | None = None,
) -> Tensor:
    r"""
    Mean rotation quaternions.

    Parameters
    ----------
    input : Tensor, shape=(..., 4)
        Rotation quaternions. Rotation quaternions are normalized to unit norm.

    weight : Tensor, shape=(..., 4), optional
        Relative importance of rotation quaternions.

    Returns
    -------
    output : Tensor, shape=(..., 4)
        Rotation quaternions mean.
    """
    if weight is None:
        weight = torch.ones(input.shape[0])

    _, output = torch.linalg.eigh((input.T * weight) @ input)

    output = output[:, -1]

    output = torch.unsqueeze(output, dim=0)

    return output
