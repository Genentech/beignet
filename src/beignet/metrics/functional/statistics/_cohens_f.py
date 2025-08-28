"""Cohen's f effect size functional metric."""

import torch
from torch import Tensor

import beignet.statistics


def cohens_f(
    groups: list[Tensor],
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Compute Cohen's f effect size from multiple sample groups.

    Cohen's f is defined as the ratio of the standard deviation of the group means
    to the within-groups standard deviation.

    Parameters
    ----------
    groups : list[Tensor]
        List of sample groups, each of shape (..., N_i) where N_i is the sample size for group i.
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        The Cohen's f values.
    """
    if len(groups) < 2:
        raise ValueError("At least two groups are required for Cohen's f")

    # Compute group means
    group_means = torch.stack([group.mean(dim=-1) for group in groups], dim=-1)

    # Compute overall mean across groups (weighted by sample sizes)
    sample_sizes = torch.tensor(
        [group.shape[-1] for group in groups],
        dtype=group_means.dtype,
        device=group_means.device,
    )
    total_sample_size = sample_sizes.sum()
    weights = sample_sizes / total_sample_size
    overall_mean = (group_means * weights).sum(dim=-1, keepdim=True)

    # Compute standard deviation of group means (between-groups variability)
    between_groups_variance = ((group_means - overall_mean) ** 2 * weights).sum(dim=-1)
    between_groups_std = torch.sqrt(between_groups_variance)

    # Compute pooled within-groups standard deviation
    within_groups_variances = []
    for group in groups:
        # Compute within-group variance using unbiased estimator
        within_var = torch.var(group, dim=-1, unbiased=True)
        within_groups_variances.append(within_var)

    # Pool within-groups variances (weighted by degrees of freedom)
    dofs = torch.tensor(
        [group.shape[-1] - 1 for group in groups],
        dtype=group_means.dtype,
        device=group_means.device,
    )
    total_dof = dofs.sum()

    pooled_within_variance = (
        sum(dof * var for dof, var in zip(dofs, within_groups_variances, strict=False))
        / total_dof
    )
    pooled_within_std = torch.sqrt(pooled_within_variance)

    # Use the statistics function to compute Cohen's f
    result = beignet.statistics.cohens_f(between_groups_std, pooled_within_std)

    if out is not None:
        out.copy_(result)
        return out

    return result
