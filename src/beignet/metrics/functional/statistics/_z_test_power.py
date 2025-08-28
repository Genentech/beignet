"""Z-test power functional metric."""

import torch
from torch import Tensor

import beignet.statistics


def z_test_power(
    preds: Tensor,
    target: Tensor,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> Tensor:
    """
    Compute statistical power for z-test given two groups.

    Parameters
    ----------
    preds : Tensor
        Predictions or first group samples.
    target : Tensor
        Targets or second group samples.
    alpha : float, default 0.05
        Type I error rate.
    alternative : str, default "two-sided"
        Type of alternative hypothesis.

    Returns
    -------
    Tensor
        Statistical power.
    """
    # Compute effect size from samples
    mean1 = torch.mean(preds, dim=-1)
    mean2 = torch.mean(target, dim=-1)

    var1 = torch.var(preds, dim=-1, unbiased=True)
    var2 = torch.var(target, dim=-1, unbiased=True)
    n1 = preds.shape[-1]
    n2 = target.shape[-1]

    pooled_std = torch.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    effect_size = (mean1 - mean2) / pooled_std
    sample_size = torch.tensor(min(n1, n2), dtype=preds.dtype, device=preds.device)

    return beignet.statistics.z_test_power(
        effect_size,
        sample_size,
        alpha=alpha,
        alternative=alternative,
    )
