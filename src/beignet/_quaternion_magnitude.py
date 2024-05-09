import torch
from torch import Tensor


def quaternion_magnitude(input: Tensor, canonical=False) -> Tensor:
    r"""
    Rotation quaternion magnitudes.

    Parameters
    ----------
    input : Tensor, shape=(..., 4)
        Rotation quaternions.

    Returns
    -------
    output : Tensor, shape=(...)
        Angles in radians. Magnitudes will be in the range :math:`[0, \pi]`.
    """
    output = torch.empty(
        input.shape[0],
        dtype=input.dtype,
        layout=input.layout,
        device=input.device,
        requires_grad=input.requires_grad,
    )

    for j in range(input.shape[0]):
        a = input[j, 0]
        b = input[j, 1]
        c = input[j, 2]
        d = input[j, 3]

        x = torch.atan2(torch.sqrt(a**2 + b**2 + c**2), torch.abs(d))

        output[j] = x * 2.0

    return output
