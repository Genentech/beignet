"""Functional effect size metrics."""

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


def hedges_g(group1: Tensor, group2: Tensor) -> Tensor:
    """
    Compute Hedges' g effect size between two groups.

    Parameters
    ----------
    group1 : Tensor
        Samples from the first group.
    group2 : Tensor
        Samples from the second group.

    Returns
    -------
    Tensor
        Hedges' g effect size.

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics.functional.statistics import hedges_g
    >>> group1 = torch.randn(20)
    >>> group2 = torch.randn(20) + 0.5
    >>> effect_size = hedges_g(group1, group2)
    """
    cohens_d_value = cohens_d(group1, group2, pooled=True)
    n1 = group1.shape[-1]
    n2 = group2.shape[-1]
    df = n1 + n2 - 2
    correction_factor = 1 - (3 / (4 * df - 1))
    return cohens_d_value * correction_factor


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


def cohens_f_squared(groups: list[Tensor]) -> Tensor:
    """
    Compute Cohen's f² effect size for ANOVA.

    Parameters
    ----------
    groups : list of Tensor
        List of sample tensors for each group.

    Returns
    -------
    Tensor
        Cohen's f² effect size.

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics.functional.statistics import cohens_f_squared
    >>> groups = [torch.randn(20), torch.randn(20) + 0.5, torch.randn(20) + 1.0]
    >>> effect_size = cohens_f_squared(groups)
    """
    f_value = cohens_f(groups)
    return f_value**2


def cramers_v(observed: Tensor, expected: Tensor) -> Tensor:
    """
    Compute Cramer's V effect size for chi-square test.

    Parameters
    ----------
    observed : Tensor
        Observed frequencies in contingency table.
    expected : Tensor
        Expected frequencies in contingency table.

    Returns
    -------
    Tensor
        Cramer's V effect size.

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics.functional.statistics import cramers_v
    >>> observed = torch.tensor([[10, 20], [30, 40]], dtype=torch.float)
    >>> expected = torch.tensor([[15, 15], [35, 35]], dtype=torch.float)
    >>> effect_size = cramers_v(observed, expected)
    """
    # Calculate chi-square statistic
    chi_square = torch.sum((observed - expected) ** 2 / expected)

    # Get dimensions
    n = torch.sum(observed)
    min_dim = min(observed.shape[-2] - 1, observed.shape[-1] - 1)

    return torch.sqrt(chi_square / (n * min_dim))


def phi_coefficient(observed: Tensor, expected: Tensor) -> Tensor:
    """
    Compute Phi coefficient for 2x2 contingency table.

    Parameters
    ----------
    observed : Tensor
        Observed frequencies in 2x2 contingency table.
    expected : Tensor
        Expected frequencies in 2x2 contingency table.

    Returns
    -------
    Tensor
        Phi coefficient.

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics.functional.statistics import phi_coefficient
    >>> observed = torch.tensor([[10, 20], [30, 40]], dtype=torch.float)
    >>> expected = torch.tensor([[15, 15], [35, 35]], dtype=torch.float)
    >>> effect_size = phi_coefficient(observed, expected)
    """
    if observed.shape[-2] != 2 or observed.shape[-1] != 2:
        raise ValueError("Phi coefficient requires a 2x2 contingency table")

    # Calculate chi-square statistic
    chi_square = torch.sum((observed - expected) ** 2 / expected)
    n = torch.sum(observed)

    return torch.sqrt(chi_square / n)
