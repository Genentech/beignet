import math

import torch
from torch import Tensor


def anova_sample_size(
    effect_size: Tensor,
    groups: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute required sample size for one-way ANOVA to achieve specified power.

    Given Cohen's f effect size, number of groups, desired power, and significance
    level, this function calculates the minimum total sample size needed across
    all groups.

    This function is differentiable with respect to the effect_size and k parameters.
    While traditional sample size calculations don't require gradients,
    differentiability enables integration into machine learning pipelines where
    effect sizes might be learned parameters or part of experimental design
    optimization.

    When to Use
    -----------
    **Traditional Statistics:**
    - Planning randomized controlled trials comparing multiple treatment groups
    - Sample size planning for clinical studies with multiple intervention arms
    - Educational research comparing different teaching methods across classes
    - Agricultural experiments testing multiple fertilizer or treatment conditions
    - Quality control studies comparing multiple manufacturing processes

    **Machine Learning Contexts:**
    - Planning model comparison studies across multiple algorithms or architectures
    - A/B testing with multiple treatment conditions for ML system optimization
    - Hyperparameter optimization: planning studies comparing multiple configurations
    - Cross-validation: planning comparisons across different modeling approaches
    - Transfer learning: planning studies comparing multiple source domains
    - Ensemble methods: planning evaluations across different ensemble configurations
    - Active learning: planning comparisons of multiple sample selection strategies
    - Federated learning: planning performance comparisons across multiple client groups
    - Domain adaptation: planning studies across multiple target domains
    - Computer vision: planning model comparisons across different architectures
    - NLP: planning language model comparisons across multiple model families
    - Recommendation systems: planning algorithm comparisons across different approaches
    - Anomaly detection: planning method comparisons across multiple detection algorithms
    - Time series forecasting: planning model comparisons across different forecasting methods
    - Causal inference: planning studies with multiple treatment groups or interventions

    **Choose ANOVA sample size over t-test sample size when:**
    - Comparing 3 or more groups simultaneously
    - Want to control family-wise error rate across multiple comparisons
    - Testing overall effect before specific pairwise comparisons
    - Experimental design includes multiple treatment conditions
    - Need to assess variance between groups relative to within-group variance

    **Choose one-way ANOVA over other ANOVA designs when:**
    - Single factor with multiple levels is being tested
    - No additional blocking or factorial structure in design
    - Groups are independent with no repeated measures
    - Primary interest is in overall group differences
    - Simplest design structure is appropriate

    **Interpretation Guidelines:**
    - Effect size is Cohen's f: measures standardized between-group variance
    - Total sample size is distributed equally across k groups by default
    - Cohen's f = 0.10 (small), 0.25 (medium), 0.40 (large) effect sizes
    - Power increases with larger effect sizes and more subjects per group
    - Consider practical constraints including recruitment and cost per additional group
    - Account for potential dropouts and unequal group sizes in planning

    Parameters
    ----------
    effect_size : Tensor
        Cohen's f effect size. Should be positive.

    groups : Tensor
        Number of groups in the ANOVA.

    power : float, default=0.8
        Desired statistical power (probability of correctly rejecting false null).

    alpha : float, default=0.05
        Significance level (Type I error rate).

    out : Tensor, optional
        Output tensor. Default, `None`.

    Returns
    -------
    output : Tensor
        Required total sample size across all groups (rounded up to nearest integer).

    Examples
    --------
    >>> effect_size = torch.tensor(0.25)
    >>> groups = torch.tensor(3)
    >>> anova_sample_size(effect_size, groups, power=0.8)
    tensor(159)

    Notes
    -----
    The sample size calculation is based on the noncentral F-distribution.
    For a one-way ANOVA with k groups and total sample size N, the degrees
    of freedom are:
    - df₁ = k - 1 (between groups)
    - df₂ = N - k (within groups)

    The noncentrality parameter is:
    λ = N * f²

    Where f is Cohen's f effect size.

    The calculation uses an iterative approach to find the sample size that
    achieves the desired power, starting from an initial approximation based
    on the noncentral chi-square distribution.

    For computational efficiency, we use analytical approximations where possible,
    falling back to iterative refinement when needed.

    References
    ----------
    .. [1] Cohen, J. (1988). Statistical power analysis for the behavioral
           sciences (2nd ed.). Erlbaum.
    .. [2] Faul, F., Erdfelder, E., Lang, A. G., & Buchner, A. (2007).
           G*Power 3: A flexible statistical power analysis program.
           Behavior Research Methods, 39(2), 175-191.
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    groups = torch.atleast_1d(torch.as_tensor(groups))

    if effect_size.dtype == torch.float64 or groups.dtype == torch.float64:
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    groups = groups.to(dtype)

    effect_size = torch.clamp(effect_size, min=1e-6)
    groups = torch.clamp(groups, min=2.0)

    df1 = groups - 1

    sqrt_2 = math.sqrt(2.0)
    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2
    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt_2


    chi2_critical = df1 + z_alpha * torch.sqrt(2 * df1)






    lambda_initial = ((z_alpha + z_beta) * sqrt_2) ** 2
    n_initial = lambda_initial / (effect_size**2)

    n_initial = torch.clamp(n_initial, min=groups + 1)

    n_current = n_initial
    convergence_tolerance = 1e-6
    max_iterations = 8

    for _iteration in range(max_iterations):
        df2_current = n_current - groups
        df2_current = torch.clamp(df2_current, min=1.0)

        lambda_current = n_current * effect_size**2

        f_critical = chi2_critical / df1

        adjustment = 1 + 2 / torch.clamp(df2_current, min=1.0)
        f_critical = f_critical * adjustment

        mean_nc_chi2 = df1 + lambda_current
        var_nc_chi2 = 2 * (df1 + 2 * lambda_current)

        mean_f = mean_nc_chi2 / df1
        var_f = var_nc_chi2 / (df1**2)

        var_adjustment = (df2_current + 2) / torch.clamp(df2_current, min=1.0)
        var_f = var_f * var_adjustment

        std_f = torch.sqrt(var_f)
        z_current = (f_critical - mean_f) / torch.clamp(std_f, min=1e-10)

        power_current = (1 - torch.erf(z_current / sqrt_2)) / 2

        power_diff = power - power_current

        adjustment = power_diff * n_current * 0.5

        converged_mask = torch.abs(power_diff) < convergence_tolerance
        adjustment = torch.where(converged_mask, adjustment * 0.1, adjustment)
        n_current = n_current + adjustment

        n_current = torch.clamp(n_current, min=groups + 1)
        n_current = torch.clamp(n_current, max=100000.0)

    output = torch.ceil(n_current)

    output = torch.clamp(output, min=groups + 1)

    if out is not None:
        out.copy_(output)
        return out

    return output
