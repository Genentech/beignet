"""Functional power analysis metrics."""

import torch
from torch import Tensor

import beignet.statistics


def t_test_power(
    preds: Tensor,
    target: Tensor,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> Tensor:
    """
    Compute statistical power for t-test given two groups.

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

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics.functional.statistics import t_test_power
    >>> group1 = torch.randn(20)
    >>> group2 = torch.randn(20) + 0.5
    >>> power = t_test_power(group1, group2)
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

    return beignet.statistics.t_test_power(
        effect_size,
        sample_size,
        alpha=alpha,
        alternative=alternative,
    )


def independent_t_test_power(
    preds: Tensor,
    target: Tensor,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> Tensor:
    """
    Compute statistical power for independent t-test given two groups.

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

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics.functional.statistics import independent_t_test_power
    >>> group1 = torch.randn(20)
    >>> group2 = torch.randn(20) + 0.5
    >>> power = independent_t_test_power(group1, group2)
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
    sample_size_1 = torch.tensor(n1, dtype=preds.dtype, device=preds.device)

    ratio = torch.tensor(n2 / n1, dtype=preds.dtype, device=preds.device)

    return beignet.statistics.independent_t_test_power(
        effect_size,
        sample_size_1,
        alpha=alpha,
        alternative=alternative,
        ratio=ratio,
    )


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

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics.functional.statistics import z_test_power
    >>> group1 = torch.randn(100)
    >>> group2 = torch.randn(100) + 0.5
    >>> power = z_test_power(group1, group2)
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


def independent_z_test_power(
    preds: Tensor,
    target: Tensor,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> Tensor:
    """
    Compute statistical power for independent z-test given two groups.

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

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics.functional.statistics import independent_z_test_power
    >>> group1 = torch.randn(100)
    >>> group2 = torch.randn(100) + 0.5
    >>> power = independent_z_test_power(group1, group2)
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
    sample_size_1 = torch.tensor(n1, dtype=preds.dtype, device=preds.device)

    ratio = torch.tensor(n2 / n1, dtype=preds.dtype, device=preds.device)

    return beignet.statistics.independent_z_test_power(
        effect_size,
        sample_size_1,
        alpha=alpha,
        alternative=alternative,
        ratio=ratio,
    )


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

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics.functional.statistics import analysis_of_variance_power
    >>> groups = [torch.randn(20), torch.randn(20) + 0.5, torch.randn(20) + 1.0]
    >>> power = analysis_of_variance_power(groups)
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


def f_test_power(preds: Tensor, target: Tensor, alpha: float = 0.05) -> Tensor:
    """
    Compute statistical power for F-test given two groups.

    Parameters
    ----------
    preds : Tensor
        Predictions or first group samples.
    target : Tensor
        Targets or second group samples.
    alpha : float, default 0.05
        Type I error rate.

    Returns
    -------
    Tensor
        Statistical power.

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics.functional.statistics import f_test_power
    >>> group1 = torch.randn(20)
    >>> group2 = torch.randn(20) * 1.5  # Different variance
    >>> power = f_test_power(group1, group2)
    """
    var1 = torch.var(preds, dim=-1, unbiased=True)
    var2 = torch.var(target, dim=-1, unbiased=True)
    effect_size = var1 / var2

    df1 = torch.tensor(preds.shape[-1] - 1, dtype=preds.dtype, device=preds.device)
    df2 = torch.tensor(target.shape[-1] - 1, dtype=preds.dtype, device=preds.device)

    return beignet.statistics.f_test_power(
        effect_size,
        df1,
        df2,
        alpha=alpha,
    )


def correlation_power(
    x: Tensor,
    y: Tensor,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> Tensor:
    """
    Compute statistical power for correlation test.

    Parameters
    ----------
    x : Tensor
        First variable samples.
    y : Tensor
        Second variable samples.
    alpha : float, default 0.05
        Type I error rate.
    alternative : str, default "two-sided"
        Type of alternative hypothesis.

    Returns
    -------
    Tensor
        Statistical power.

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics.functional.statistics import correlation_power
    >>> x = torch.randn(50)
    >>> y = 0.5 * x + torch.randn(50) * 0.5  # Correlated
    >>> power = correlation_power(x, y)
    """
    # Calculate correlation coefficient
    x_centered = x - torch.mean(x, dim=-1, keepdim=True)
    y_centered = y - torch.mean(y, dim=-1, keepdim=True)

    numerator = torch.sum(x_centered * y_centered, dim=-1)
    denominator = torch.sqrt(
        torch.sum(x_centered**2, dim=-1) * torch.sum(y_centered**2, dim=-1),
    )
    correlation = numerator / denominator

    sample_size = torch.tensor(x.shape[-1], dtype=x.dtype, device=x.device)

    return beignet.statistics.correlation_power(
        correlation,
        sample_size,
        alpha=alpha,
        alternative=alternative,
    )


def proportion_power(
    successes: Tensor,
    trials: Tensor,
    null_proportion: float = 0.5,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> Tensor:
    """
    Compute statistical power for proportion test.

    Parameters
    ----------
    successes : Tensor
        Number of successes.
    trials : Tensor
        Number of trials.
    null_proportion : float, default 0.5
        Null hypothesis proportion.
    alpha : float, default 0.05
        Type I error rate.
    alternative : str, default "two-sided"
        Type of alternative hypothesis.

    Returns
    -------
    Tensor
        Statistical power.

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics.functional.statistics import proportion_power
    >>> successes = torch.tensor([60])
    >>> trials = torch.tensor([100])
    >>> power = proportion_power(successes, trials)
    """
    observed_proportion = successes / trials

    return beignet.statistics.proportion_power(
        observed_proportion,
        trials,
        null_proportion=null_proportion,
        alpha=alpha,
        alternative=alternative,
    )


def proportion_two_sample_power(
    successes1: Tensor,
    trials1: Tensor,
    successes2: Tensor,
    trials2: Tensor,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> Tensor:
    """
    Compute statistical power for two-sample proportion test.

    Parameters
    ----------
    successes1 : Tensor
        Number of successes in group 1.
    trials1 : Tensor
        Number of trials in group 1.
    successes2 : Tensor
        Number of successes in group 2.
    trials2 : Tensor
        Number of trials in group 2.
    alpha : float, default 0.05
        Type I error rate.
    alternative : str, default "two-sided"
        Type of alternative hypothesis.

    Returns
    -------
    Tensor
        Statistical power.

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics.functional.statistics import proportion_two_sample_power
    >>> successes1, trials1 = torch.tensor([60]), torch.tensor([100])
    >>> successes2, trials2 = torch.tensor([45]), torch.tensor([100])
    >>> power = proportion_two_sample_power(successes1, trials1, successes2, trials2)
    """
    proportion1 = successes1 / trials1
    proportion2 = successes2 / trials2

    return beignet.statistics.proportion_two_sample_power(
        proportion1,
        trials1,
        proportion2,
        trials2,
        alpha=alpha,
        alternative=alternative,
    )


def chi_squared_goodness_of_fit_power(
    observed: Tensor,
    expected: Tensor,
    alpha: float = 0.05,
) -> Tensor:
    """
    Compute statistical power for chi-square goodness of fit test.

    Parameters
    ----------
    observed : Tensor
        Observed frequencies.
    expected : Tensor
        Expected frequencies.
    alpha : float, default 0.05
        Type I error rate.

    Returns
    -------
    Tensor
        Statistical power.

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics.functional.statistics import chi_squared_goodness_of_fit_power
    >>> observed = torch.tensor([10, 20, 15, 25], dtype=torch.float)
    >>> expected = torch.tensor([17.5, 17.5, 17.5, 17.5], dtype=torch.float)
    >>> power = chi_squared_goodness_of_fit_power(observed, expected)
    """
    effect_size = torch.sqrt(
        torch.sum((observed - expected) ** 2 / expected) / torch.sum(expected),
    )
    sample_size = torch.sum(observed)
    degrees_of_freedom = torch.tensor(
        observed.shape[-1] - 1,
        dtype=observed.dtype,
        device=observed.device,
    )

    return beignet.statistics.chi_squared_goodness_of_fit_power(
        effect_size,
        sample_size,
        degrees_of_freedom,
        alpha=alpha,
    )


def chi_squared_independence_power(
    observed: Tensor,
    expected: Tensor,
    alpha: float = 0.05,
) -> Tensor:
    """
    Compute statistical power for chi-square independence test.

    Parameters
    ----------
    observed : Tensor
        Observed frequencies in contingency table.
    expected : Tensor
        Expected frequencies in contingency table.
    alpha : float, default 0.05
        Type I error rate.

    Returns
    -------
    Tensor
        Statistical power.

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics.functional.statistics import chi_squared_independence_power
    >>> observed = torch.tensor([[10, 20], [30, 40]], dtype=torch.float)
    >>> expected = torch.tensor([[15, 15], [35, 35]], dtype=torch.float)
    >>> power = chi_squared_independence_power(observed, expected)
    """
    effect_size = torch.sqrt(
        torch.sum((observed - expected) ** 2 / expected) / torch.sum(expected),
    )
    sample_size = torch.sum(observed)
    rows = torch.tensor(
        observed.shape[-2],
        dtype=observed.dtype,
        device=observed.device,
    )
    columns = torch.tensor(
        observed.shape[-1],
        dtype=observed.dtype,
        device=observed.device,
    )

    return beignet.statistics.chi_squared_independence_power(
        effect_size,
        sample_size,
        rows,
        columns,
        alpha=alpha,
    )
