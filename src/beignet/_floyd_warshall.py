import torch
from torch import Tensor


def floyd_warshall(
    input: Tensor,
    directed: bool = True,
    unweighted: bool = False,
):
    r"""
    ...

    Parameters
    ----------
    input : Tensor, shape=(..., N, N)
        ...

    directed : bool
        If `False`, symmetrizes `input`. Default, `True`.

    unweighted : bool
        If `True`, distance of non-zero connections is 1. Default, `False`.

    Returns
    -------
    output : Tensor, shape=(..., N, N)
        ...
    """
    output = input.clone()

    if not directed:
        output = 0.5 * (output + output.transpose(-1, -2))

    if unweighted:
        output = torch.where(
            output != 0,
            torch.ones_like(output),
            torch.zeros_like(output),
        )

    n = output.shape[-1]

    eye = torch.eye(n, device=output.device, dtype=output.dtype)

    eye = torch.expand_copy(eye, output.shape)

    output[((output == 0) & (~eye.bool()))] = torch.inf

    output = torch.where(
        eye.to(dtype=torch.bool),
        torch.zeros_like(output),
        output,
    )

    for k in range(n):
        a = torch.unsqueeze(output[..., :, k], dim=-1)
        b = torch.unsqueeze(output[..., k, :], dim=-2)

        output = torch.minimum(output, a + b)

    return output
