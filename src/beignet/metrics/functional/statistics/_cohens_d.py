"""Cohen's d effect size functional metric."""

import torch
from torch import Tensor


def cohens_d(group1: Tensor, group2: Tensor, pooled: bool = True) -> Tensor:
    """
    Compute Cohen's d effect size between two groups.

    Parameters
    ----------
    group1 : Tensor
        Samples from the first group.
    group2 : Tensor
        Samples from the second group.
    pooled : bool, default True
        Whether to use pooled standard deviation for the denominator.

    Returns
    -------
    Tensor
        Cohen's d effect size.

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics.functional.statistics import cohens_d
    >>> group1 = torch.randn(20)
    >>> group2 = torch.randn(20) + 0.5
    >>> effect_size = cohens_d(group1, group2)
    """
    mean1 = torch.mean(group1, dim=-1, keepdim=True)
    mean2 = torch.mean(group2, dim=-1, keepdim=True)

    if pooled:
        var1 = torch.var(group1, dim=-1, unbiased=True, keepdim=True)
        var2 = torch.var(group2, dim=-1, unbiased=True, keepdim=True)
        n1 = group1.shape[-1]
        n2 = group2.shape[-1]
        pooled_std = torch.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return (mean1 - mean2).squeeze(-1) / pooled_std.squeeze(-1)
    else:
        std1 = torch.std(group1, dim=-1, unbiased=True, keepdim=True)
        return (mean1 - mean2).squeeze(-1) / std1.squeeze(-1)
