import math

import torch
from torch import Tensor


def independent_t_test_sample_size(
    effect_size: Tensor,
    ratio: Tensor | None = None,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute required sample size for independent samples t-tests.

    Given the effect size, sample size ratio, desired power, and significance
    level, this function calculates the minimum sample size for group 1 needed
    to achieve the specified power for an independent samples t-test.

    This function is differentiable with respect to effect_size and ratio parameters.
    While traditional sample size calculations don't require gradients,
    differentiability enables integration into machine learning pipelines where
    effect sizes might be learned parameters or part of experimental design
    optimization.

    When to Use
    -----------
    **Traditional Statistics:**
    - Planning randomized controlled trials with two independent groups
    - Designing case-control studies comparing different populations
    - Sample size planning for clinical trials with treatment and control groups
    - A/B testing in marketing research with independent user groups
    - Survey research comparing different demographic groups

    **Machine Learning Contexts:**
    - Planning model comparison studies with independent test sets
    - A/B testing ML models with different user populations
    - Experimental design for comparing algorithms on different datasets
    - Planning studies to compare model performance across different domains
    - Transfer learning: comparing performance between source and target domains
    - Hyperparameter optimization: comparing configurations with independent data
    - Ensemble methods: comparing performance across different training sets
    - Active learning: comparing selection strategies with separate data pools
    - Federated learning: comparing performance across different client populations
    - Domain adaptation: planning studies for cross-domain model performance
    - Computer vision: comparing models on independent image datasets
    - NLP: comparing text classification models on different corpora
    - Recommendation systems: comparing algorithms across different user groups
    - Anomaly detection: comparing detection methods on independent datasets
    - Causal inference: planning studies with treatment and control groups

    **Choose independent t-test sample size over paired t-test when:**
    - Groups are naturally independent (different subjects/datasets)
    - No meaningful pairing or matching is possible
    - Between-group comparisons are of primary interest
    - Groups have different characteristics that can't be matched
    - Randomized assignment to groups is used

    **Choose independent t-test over other tests when:**
    - Comparing means of two continuous variables
    - Data is approximately normally distributed within groups
    - Independent observations within and between groups
    - Equal variances assumed (use Welch's t-test if unequal)
    - Sample sizes are moderate to large (n ≥ 30 per group)

    **Interpretation Guidelines:**
    - Effect size is Cohen's d: (μ₁ - μ₂) / σ_pooled
    - Sample size ratio allows unequal group sizes (n₂ = ratio × n₁)
    - Power increases with larger effect sizes and sample sizes
    - Consider practical constraints (cost, time, recruitment feasibility)
    - Account for potential dropout rates in planning
    - Small effect sizes require substantially larger samples

    Parameters
    ----------
    effect_size : Tensor
        Standardized effect size (Cohen's d). This is the difference between
        the two group means divided by the pooled standard deviation:
        d = (μ₁ - μ₂) / σ_pooled. Should be positive.
    ratio : Tensor, default=1.0
        Ratio of sample size for group 2 relative to group 1.
        Group 2 sample size = nobs1 * ratio.
    power : float, default=0.8
        Desired statistical power (probability of correctly rejecting false null).
    alpha : float, default=0.05
        Significance level (Type I error rate).
    alternative : str, default="two-sided"
        Type of alternative hypothesis. Either "two-sided", "larger", or "smaller".

    Returns
    -------
    output : Tensor
        Required sample size for group 1 (rounded up to nearest integer).
        Group 2 sample size can be calculated as output * ratio.

    Examples
    --------
    >>> effect_size = torch.tensor(0.5)
    >>> independent_t_test_sample_size(effect_size, power=0.8)
    tensor(64)

    Notes
    -----
    The sample size calculation is based on the noncentral t-distribution.
    For an independent samples t-test with effect size d, the noncentrality
    parameter is:

    δ = d / √(1/n₁ + 1/n₂)

    Where n₁ and n₂ are the sample sizes for groups 1 and 2.

    The calculation starts with a normal approximation:

    n₁ ≈ ((z_α + z_β) / d)² * (1 + 1/r) / 2

    Where r is the ratio n₂/n₁, and z_α, z_β are critical values.

    For computational efficiency, we use analytical approximations where possible,
    falling back to iterative refinement to account for finite degrees of freedom
    effects.

    References
    ----------
    .. [1] Cohen, J. (1988). Statistical power analysis for the behavioral
           sciences (2nd ed.). Erlbaum.
    .. [2] Aberson, C. L. (2010). Applied power analysis for the behavioral
           sciences. Routledge.
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    if ratio is None:
        ratio = torch.tensor(1.0)
    else:
        ratio = torch.atleast_1d(torch.as_tensor(ratio))

    if effect_size.dtype == torch.float64 or ratio.dtype == torch.float64:
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    ratio = ratio.to(dtype)

    effect_size = torch.clamp(effect_size, min=1e-6)
    ratio = torch.clamp(ratio, min=0.1, max=10.0)

    sqrt_2 = math.sqrt(2.0)

    if alternative == "two-sided":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt_2
    else:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2

    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt_2

    variance_factor = (1 + 1 / ratio) / 2
    sample_size_group_1_initial = (
        (z_alpha + z_beta) / effect_size
    ) ** 2 * variance_factor

    sample_size_group_1_initial = torch.clamp(sample_size_group_1_initial, min=2.0)

    sample_size_group_1_current = sample_size_group_1_initial
    convergence_tolerance = 1e-6
    max_iterations = 10

    for _iteration in range(max_iterations):
        sample_size_group_2_current = sample_size_group_1_current * ratio
        total_n = sample_size_group_1_current + sample_size_group_2_current
        df_current = total_n - 2
        df_current = torch.clamp(df_current, min=1.0)

        se_factor = torch.sqrt(
            1 / sample_size_group_1_current + 1 / sample_size_group_2_current
        )

        ncp_current = effect_size / se_factor

        if alternative == "two-sided":
            t_critical = z_alpha * torch.sqrt(1 + 1 / (2 * df_current))
        else:
            t_critical = z_alpha * torch.sqrt(1 + 1 / (2 * df_current))

        var_nct = torch.where(
            df_current > 2,
            (df_current + ncp_current**2) / (df_current - 2),
            1 + ncp_current**2 / (2 * df_current),
        )
        std_nct = torch.sqrt(var_nct)

        if alternative == "two-sided":
            z_upper = (t_critical - ncp_current) / torch.clamp(std_nct, min=1e-10)
            z_lower = (-t_critical - ncp_current) / torch.clamp(std_nct, min=1e-10)
            power_current = (1 - torch.erf(z_upper / sqrt_2)) / 2 + (
                1 - torch.erf(-z_lower / sqrt_2)
            ) / 2
        elif alternative == "larger":
            z_score = (t_critical - ncp_current) / torch.clamp(std_nct, min=1e-10)
            power_current = (1 - torch.erf(z_score / sqrt_2)) / 2
        else:
            z_score = (-t_critical - (-ncp_current)) / torch.clamp(std_nct, min=1e-10)
            power_current = (1 - torch.erf(-z_score / sqrt_2)) / 2

        power_current = torch.clamp(power_current, 0.01, 0.99)

        power_diff = power - power_current

        adjustment = (
            power_diff
            * sample_size_group_1_current
            / (2 * torch.clamp(power_current * (1 - power_current), min=0.01))
        )

        converged_mask = torch.abs(power_diff) < convergence_tolerance
        adjustment = torch.where(converged_mask, adjustment * 0.1, adjustment)
        sample_size_group_1_current = sample_size_group_1_current + adjustment

        sample_size_group_1_current = torch.clamp(
            sample_size_group_1_current, min=2.0, max=100000.0
        )

    output = torch.ceil(sample_size_group_1_current)

    output = torch.clamp(output, min=2.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
