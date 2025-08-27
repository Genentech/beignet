import torch
from torch import Tensor


def cohens_f(
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

    epsilon = torch.finfo(dtype).eps

    output = torch.abs(other) < epsilon
    output = torch.where(output, epsilon, other)
    output = torch.std(input, dim=-1, correction=0) / output

    if out is not None:
        out.copy_(output)

        return out

    return output
