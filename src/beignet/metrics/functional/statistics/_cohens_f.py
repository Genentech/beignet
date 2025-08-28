"""Cohen's f effect size functional metric."""

import torch
from torch import Tensor


def cohens_f(groups: list[Tensor]) -> Tensor:
    """
    Compute Cohen's f effect size for ANOVA.

    Parameters
    ----------
    groups : list of Tensor
        List of sample tensors for each group.

    Returns
    -------
    Tensor
        Cohen's f effect size.

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics.functional.statistics import cohens_f
    >>> groups = [torch.randn(20), torch.randn(20) + 0.5, torch.randn(20) + 1.0]
    >>> effect_size = cohens_f(groups)
    """
    # Calculate means and overall mean
    group_means = [torch.mean(group, dim=-1) for group in groups]
    overall_mean = torch.mean(torch.cat(groups, dim=-1), dim=-1)

    # Calculate between-group variance
    n_groups = len(groups)
    group_sizes = [group.shape[-1] for group in groups]
    total_n = sum(group_sizes)

    between_ss = sum(
        n * (mean - overall_mean) ** 2
        for n, mean in zip(group_sizes, group_means, strict=False)
    )
    between_ms = between_ss / (n_groups - 1)

    # Calculate within-group variance
    within_ss = sum(
        torch.sum((group - mean) ** 2, dim=-1)
        for group, mean in zip(groups, group_means, strict=False)
    )
    within_ms = within_ss / (total_n - n_groups)

    return torch.sqrt(between_ms / within_ms)
