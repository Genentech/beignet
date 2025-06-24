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

    a, b, c, d = torch.unbind(input, dim=-1)
    return 2 * torch.atan2(torch.sqrt(a**2 + b**2 + c**2), torch.abs(d))
