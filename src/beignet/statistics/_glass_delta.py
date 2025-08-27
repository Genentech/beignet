import torch
from torch import Tensor


def glass_delta(
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

    output = (torch.mean(input, dim=-1) - torch.mean(other, dim=-1)) / torch.sqrt(
        torch.clamp(torch.var(other, dim=-1, correction=1), min=torch.finfo(dtype).eps),
    )

    if out is not None:
        out.copy_(output)

        return out

    return output
