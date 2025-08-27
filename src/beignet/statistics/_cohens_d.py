import torch
from torch import Tensor


def cohens_d(
    input: Tensor,
    other: Tensor,
    pooled: bool = True,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    input : Tensor

    other : Tensor

    pooled : bool, default True

    out : Tensor | None

    Returns
    -------
    Tensor
    """

    input_mean = torch.mean(input, dim=-1)
    other_mean = torch.mean(other, dim=-1)

    if pooled:
        input_variance = torch.var(input, dim=-1)
        other_variance = torch.var(other, dim=-1)

        input_variance = (input.shape[-1] - 1) * input_variance
        other_variance = (other.shape[-1] - 1) * other_variance

        variance = input_variance + other_variance

        output = torch.sqrt(variance / (input.shape[-1] + other.shape[-1] - 2))
    else:
        output = torch.std(input, dim=-1, unbiased=True)

    output = (input_mean - other_mean) / output

    if out is not None:
        out.copy_(output)

        return out

    return output
