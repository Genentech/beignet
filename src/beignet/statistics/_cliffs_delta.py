import torch
from torch import Tensor


def cliffs_delta(
    input: Tensor,
    other: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    input : Tensor

    other : Tensor

    out : Tensor | None

    Returns
    -------
    Tensor
    """

    input = torch.atleast_1d(input)
    other = torch.atleast_1d(other)

    dtype = torch.promote_types(input.dtype, other.dtype)

    input = input.to(dtype)
    other = other.to(dtype)

    difference = torch.unsqueeze(input, dim=-1) - torch.unsqueeze(other, dim=-2)

    input_other = torch.sigmoid(100.0 * (+difference))
    other_input = torch.sigmoid(100.0 * (-difference))

    input_other = torch.sum(input_other, dim=[-2, -1])
    other_input = torch.sum(other_input, dim=[-2, -1])

    input = torch.tensor(input.shape[-1], dtype=dtype)
    other = torch.tensor(other.shape[-1], dtype=dtype)

    output = (input_other - other_input) / (input * other)

    if out is not None:
        out.copy_(output)

        return out

    return output
