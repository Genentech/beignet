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

    input_mean = torch.mean(input, dim=-1)
    other_mean = torch.mean(other, dim=-1)

    other_variance = torch.var(other, dim=-1, correction=1)

    output = torch.clamp(other_variance, min=torch.finfo(dtype).eps)
    output = (input_mean - other_mean) / torch.sqrt(output)

    if out is not None:
        out.copy_(output)

        return out

    return output
