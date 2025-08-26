import torch
from torch import Tensor


def cohens_d(
    group1: Tensor,
    group2: Tensor,
    pooled: bool = True,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    group1 : Tensor
        Group1 parameter.
    group2 : Tensor
        Group2 parameter.
    pooled : bool, default True
        Pooled parameter.
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        Computed statistic.
    """

    if pooled:
        result = (torch.mean(group1, dim=-1) - torch.mean(group2, dim=-1)) / torch.sqrt(
            (
                (group1.shape[-1] - 1) * torch.var(group1, dim=-1, unbiased=True)
                + (group2.shape[-1] - 1) * torch.var(group2, dim=-1, unbiased=True)
            )
            / (group1.shape[-1] + group2.shape[-1] - 2),
        )
    else:
        result = (torch.mean(group1, dim=-1) - torch.mean(group2, dim=-1)) / torch.std(
            group1,
            dim=-1,
            unbiased=True,
        )

    if out is not None:
        out.copy_(result)
        return out

    return result
