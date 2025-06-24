import torch
from torch import Tensor


def canonicalize_quaternion(input: Tensor):
    """Canonicalize the input quaternion

    Parameters
    ----------
    input: Tensor, shape=(...,4)
        Rotation quaternion

    Returns
    -------
    output: Tensor, shape=(...,4)
        Canonicalized quaternion

    The canonicalized quaternion is chosen from :math:`{q, -q}`
    such that the :math:`w` term is positive.
    If the :math:`w` term is :math:`0`, then the rotation quaternion is
    chosen such that the first non-zero term of the :math:`x`, :math:`y`,
    and :math:`z` terms is positive.
    """

    a, b, c, d = torch.unbind(input, dim=-1)
    mask = (d == 0) & ((a == 0) & ((b == 0) & (c < 0) | (b < 0)) | (a < 0)) | (d < 0)

    return torch.where(mask[..., None], -input, input)
