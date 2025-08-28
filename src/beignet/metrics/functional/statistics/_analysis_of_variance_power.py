"""Analysis of variance power functional metric."""

import torch
from torch import Tensor

import beignet.statistics


def analysis_of_variance_power(groups: list[Tensor], alpha: float = 0.05) -> Tensor:
    """
    Compute statistical power for ANOVA given multiple groups.

    Parameters
    ----------
    groups : list of Tensor
        List of sample tensors for each group.
    alpha : float, default 0.05
        Type I error rate.

    Returns
    -------
    Tensor
        Statistical power.
    """
    # Calculate effect size (Cohen's f)
    group_means = [torch.mean(group, dim=-1) for group in groups]
    overall_mean = torch.mean(torch.cat(groups, dim=-1), dim=-1)

    n_groups = len(groups)
    group_sizes = [group.shape[-1] for group in groups]
    total_n = sum(group_sizes)

    between_ss = sum(
        n * (mean - overall_mean) ** 2
        for n, mean in zip(group_sizes, group_means, strict=False)
    )
    between_ms = between_ss / (n_groups - 1)

    within_ss = sum(
        torch.sum((group - mean) ** 2, dim=-1)
        for group, mean in zip(groups, group_means, strict=False)
    )
    within_ms = within_ss / (total_n - n_groups)

    effect_size = torch.sqrt(between_ms / within_ms)
    sample_size = torch.tensor(
        min(group_sizes),
        dtype=groups[0].dtype,
        device=groups[0].device,
    )
    num_groups = torch.tensor(n_groups, dtype=groups[0].dtype, device=groups[0].device)

    return beignet.statistics.analysis_of_variance_power(
        effect_size,
        sample_size,
        num_groups,
        alpha=alpha,
    )
