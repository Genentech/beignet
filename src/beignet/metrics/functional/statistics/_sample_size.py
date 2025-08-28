"""Functional sample size metrics."""

import torch
from torch import Tensor

import beignet.statistics


def t_test_sample_size(
    preds: Tensor,
    target: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> Tensor:
    """
    Compute required sample size for t-test given two groups.

    Parameters
    ----------
    preds : Tensor
        Predictions or first group samples.
    target : Tensor
        Targets or second group samples.
    power : float, default 0.8
        Desired statistical power.
    alpha : float, default 0.05
        Type I error rate.
    alternative : str, default "two-sided"
        Type of alternative hypothesis.

    Returns
    -------
    Tensor
        Required sample size.

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics.functional.statistics import t_test_sample_size
    >>> group1 = torch.randn(20)
    >>> group2 = torch.randn(20) + 0.5
    >>> sample_size = t_test_sample_size(group1, group2)
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

    return beignet.statistics.t_test_sample_size(
        effect_size,
        power=power,
        alpha=alpha,
        alternative=alternative,
    )


def independent_t_test_sample_size(
    preds: Tensor,
    target: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    ratio: float = 1.0,
) -> Tensor:
    """
    Compute required sample size for independent t-test given two groups.

    Parameters
    ----------
    preds : Tensor
        Predictions or first group samples.
    target : Tensor
        Targets or second group samples.
    power : float, default 0.8
        Desired statistical power.
    alpha : float, default 0.05
        Type I error rate.
    alternative : str, default "two-sided"
        Type of alternative hypothesis.
    ratio : float, default 1.0
        Ratio of sample sizes (n2/n1).

    Returns
    -------
    Tensor
        Required sample size for group 1.

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics.functional.statistics import independent_t_test_sample_size
    >>> group1 = torch.randn(20)
    >>> group2 = torch.randn(20) + 0.5
    >>> sample_size = independent_t_test_sample_size(group1, group2)
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

    ratio_tensor = torch.tensor(ratio, dtype=preds.dtype, device=preds.device)

    return beignet.statistics.independent_t_test_sample_size(
        effect_size,
        power=power,
        alpha=alpha,
        alternative=alternative,
        ratio=ratio_tensor,
    )


def z_test_sample_size(
    preds: Tensor,
    target: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> Tensor:
    """
    Compute required sample size for z-test given two groups.

    Parameters
    ----------
    preds : Tensor
        Predictions or first group samples.
    target : Tensor
        Targets or second group samples.
    power : float, default 0.8
        Desired statistical power.
    alpha : float, default 0.05
        Type I error rate.
    alternative : str, default "two-sided"
        Type of alternative hypothesis.

    Returns
    -------
    Tensor
        Required sample size.

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics.functional.statistics import z_test_sample_size
    >>> group1 = torch.randn(100)
    >>> group2 = torch.randn(100) + 0.5
    >>> sample_size = z_test_sample_size(group1, group2)
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

    return beignet.statistics.z_test_sample_size(
        effect_size,
        power=power,
        alpha=alpha,
        alternative=alternative,
    )


def independent_z_test_sample_size(
    preds: Tensor,
    target: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    ratio: float = 1.0,
) -> Tensor:
    """
    Compute required sample size for independent z-test given two groups.

    Parameters
    ----------
    preds : Tensor
        Predictions or first group samples.
    target : Tensor
        Targets or second group samples.
    power : float, default 0.8
        Desired statistical power.
    alpha : float, default 0.05
        Type I error rate.
    alternative : str, default "two-sided"
        Type of alternative hypothesis.
    ratio : float, default 1.0
        Ratio of sample sizes (n2/n1).

    Returns
    -------
    Tensor
        Required sample size for group 1.

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics.functional.statistics import independent_z_test_sample_size
    >>> group1 = torch.randn(100)
    >>> group2 = torch.randn(100) + 0.5
    >>> sample_size = independent_z_test_sample_size(group1, group2)
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

    ratio_tensor = torch.tensor(ratio, dtype=preds.dtype, device=preds.device)

    return beignet.statistics.independent_z_test_sample_size(
        effect_size,
        power=power,
        alpha=alpha,
        alternative=alternative,
        ratio=ratio_tensor,
    )


def analysis_of_variance_sample_size(
    groups: list[Tensor],
    power: float = 0.8,
    alpha: float = 0.05,
) -> Tensor:
    """
    Compute required sample size for ANOVA given multiple groups.

    Parameters
    ----------
    groups : list of Tensor
        List of sample tensors for each group.
    power : float, default 0.8
        Desired statistical power.
    alpha : float, default 0.05
        Type I error rate.

    Returns
    -------
    Tensor
        Required sample size per group.

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics.functional.statistics import analysis_of_variance_sample_size
    >>> groups = [torch.randn(20), torch.randn(20) + 0.5, torch.randn(20) + 1.0]
    >>> sample_size = analysis_of_variance_sample_size(groups)
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
    num_groups = torch.tensor(n_groups, dtype=groups[0].dtype, device=groups[0].device)

    return beignet.statistics.analysis_of_variance_sample_size(
        effect_size,
        num_groups,
        power=power,
        alpha=alpha,
    )


def f_test_sample_size(
    preds: Tensor,
    target: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
) -> Tensor:
    """
    Compute required sample size for F-test given two groups.

    Parameters
    ----------
    preds : Tensor
        Predictions or first group samples.
    target : Tensor
        Targets or second group samples.
    power : float, default 0.8
        Desired statistical power.
    alpha : float, default 0.05
        Type I error rate.

    Returns
    -------
    Tensor
        Required degrees of freedom for numerator.

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics.functional.statistics import f_test_sample_size
    >>> group1 = torch.randn(20)
    >>> group2 = torch.randn(20) * 1.5  # Different variance
    >>> sample_size = f_test_sample_size(group1, group2)
    """
    var1 = torch.var(preds, dim=-1, unbiased=True)
    var2 = torch.var(target, dim=-1, unbiased=True)
    effect_size = var1 / var2

    df2 = torch.tensor(target.shape[-1] - 1, dtype=preds.dtype, device=preds.device)

    return beignet.statistics.f_test_sample_size(
        effect_size,
        df2,
        power=power,
        alpha=alpha,
    )


def correlation_sample_size(
    x: Tensor,
    y: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> Tensor:
    """
    Compute required sample size for correlation test.

    Parameters
    ----------
    x : Tensor
        First variable samples.
    y : Tensor
        Second variable samples.
    power : float, default 0.8
        Desired statistical power.
    alpha : float, default 0.05
        Type I error rate.
    alternative : str, default "two-sided"
        Type of alternative hypothesis.

    Returns
    -------
    Tensor
        Required sample size.

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics.functional.statistics import correlation_sample_size
    >>> x = torch.randn(50)
    >>> y = 0.5 * x + torch.randn(50) * 0.5  # Correlated
    >>> sample_size = correlation_sample_size(x, y)
    """
    # Calculate correlation coefficient
    x_centered = x - torch.mean(x, dim=-1, keepdim=True)
    y_centered = y - torch.mean(y, dim=-1, keepdim=True)

    numerator = torch.sum(x_centered * y_centered, dim=-1)
    denominator = torch.sqrt(
        torch.sum(x_centered**2, dim=-1) * torch.sum(y_centered**2, dim=-1),
    )
    correlation = numerator / denominator

    return beignet.statistics.correlation_sample_size(
        correlation,
        power=power,
        alpha=alpha,
        alternative=alternative,
    )


def proportion_sample_size(
    successes: Tensor,
    trials: Tensor,
    null_proportion: float = 0.5,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> Tensor:
    """
    Compute required sample size for proportion test.

    Parameters
    ----------
    successes : Tensor
        Number of successes.
    trials : Tensor
        Number of trials.
    null_proportion : float, default 0.5
        Null hypothesis proportion.
    power : float, default 0.8
        Desired statistical power.
    alpha : float, default 0.05
        Type I error rate.
    alternative : str, default "two-sided"
        Type of alternative hypothesis.

    Returns
    -------
    Tensor
        Required sample size.

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics.functional.statistics import proportion_sample_size
    >>> successes = torch.tensor([60])
    >>> trials = torch.tensor([100])
    >>> sample_size = proportion_sample_size(successes, trials)
    """
    observed_proportion = successes / trials

    return beignet.statistics.proportion_sample_size(
        observed_proportion,
        null_proportion=null_proportion,
        power=power,
        alpha=alpha,
        alternative=alternative,
    )


def proportion_two_sample_sample_size(
    successes1: Tensor,
    trials1: Tensor,
    successes2: Tensor,
    trials2: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    ratio: float = 1.0,
) -> Tensor:
    """
    Compute required sample size for two-sample proportion test.

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
    power : float, default 0.8
        Desired statistical power.
    alpha : float, default 0.05
        Type I error rate.
    alternative : str, default "two-sided"
        Type of alternative hypothesis.
    ratio : float, default 1.0
        Ratio of sample sizes (n2/n1).

    Returns
    -------
    Tensor
        Required sample size for group 1.

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics.functional.statistics import proportion_two_sample_sample_size
    >>> successes1, trials1 = torch.tensor([60]), torch.tensor([100])
    >>> successes2, trials2 = torch.tensor([45]), torch.tensor([100])
    >>> sample_size = proportion_two_sample_sample_size(successes1, trials1, successes2, trials2)
    """
    proportion1 = successes1 / trials1
    proportion2 = successes2 / trials2

    ratio_tensor = torch.tensor(ratio, dtype=successes1.dtype, device=successes1.device)

    return beignet.statistics.proportion_two_sample_sample_size(
        proportion1,
        proportion2,
        ratio=ratio_tensor,
        power=power,
        alpha=alpha,
        alternative=alternative,
    )


def chi_squared_goodness_of_fit_sample_size(
    observed: Tensor,
    expected: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
) -> Tensor:
    """
    Compute required sample size for chi-square goodness of fit test.

    Parameters
    ----------
    observed : Tensor
        Observed frequencies.
    expected : Tensor
        Expected frequencies.
    power : float, default 0.8
        Desired statistical power.
    alpha : float, default 0.05
        Type I error rate.

    Returns
    -------
    Tensor
        Required sample size.

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics.functional.statistics import chi_squared_goodness_of_fit_sample_size
    >>> observed = torch.tensor([10, 20, 15, 25], dtype=torch.float)
    >>> expected = torch.tensor([17.5, 17.5, 17.5, 17.5], dtype=torch.float)
    >>> sample_size = chi_squared_goodness_of_fit_sample_size(observed, expected)
    """
    effect_size = torch.sqrt(
        torch.sum((observed - expected) ** 2 / expected) / torch.sum(expected),
    )
    degrees_of_freedom = torch.tensor(
        observed.shape[-1] - 1,
        dtype=observed.dtype,
        device=observed.device,
    )

    return beignet.statistics.chi_squared_goodness_of_fit_sample_size(
        effect_size,
        degrees_of_freedom,
        power=power,
        alpha=alpha,
    )


def chi_squared_independence_sample_size(
    observed: Tensor,
    expected: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
) -> Tensor:
    """
    Compute required sample size for chi-square independence test.

    Parameters
    ----------
    observed : Tensor
        Observed frequencies in contingency table.
    expected : Tensor
        Expected frequencies in contingency table.
    power : float, default 0.8
        Desired statistical power.
    alpha : float, default 0.05
        Type I error rate.

    Returns
    -------
    Tensor
        Required sample size.

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics.functional.statistics import chi_squared_independence_sample_size
    >>> observed = torch.tensor([[10, 20], [30, 40]], dtype=torch.float)
    >>> expected = torch.tensor([[15, 15], [35, 35]], dtype=torch.float)
    >>> sample_size = chi_squared_independence_sample_size(observed, expected)
    """
    effect_size = torch.sqrt(
        torch.sum((observed - expected) ** 2 / expected) / torch.sum(expected),
    )
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

    return beignet.statistics.chi_squared_independence_sample_size(
        effect_size,
        rows,
        columns,
        power=power,
        alpha=alpha,
    )
