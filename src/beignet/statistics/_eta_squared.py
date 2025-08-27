import torch
from torch import Tensor


def eta_squared(
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

    input = torch.clamp(input, min=0.0)

    other = torch.clamp(other, min=torch.finfo(dtype).eps)

    input = torch.clamp(input, max=other)

    output = torch.clamp(input / other, 0.0, 1.0)

    if out is not None:
        out.copy_(output)

        return out

    return output
