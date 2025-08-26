import math

import torch
from torch import Tensor


def friedman_test_power(
    effect_size: Tensor,
    n_subjects: Tensor,
    n_treatments: Tensor,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Power analysis for the Friedman test.

    The Friedman test is a non-parametric alternative to repeated measures ANOVA
    for comparing three or more related groups when the assumption of normality
    is violated.

    When to Use
    -----------
    **Traditional Statistics:**
    - Repeated measures with 3+ conditions and non-normal data
    - Within-subjects designs with ordinal outcomes
    - Matched groups comparisons (e.g., matched triplets)
    - Longitudinal studies with non-parametric data

    **Machine Learning Contexts:**
    - Comparing multiple ML models on same datasets (repeated measures)
    - Cross-validation: comparing algorithms across multiple folds
    - Hyperparameter optimization: comparing parameter sets within subjects
    - Time series: comparing model performance across repeated time periods
    - A/B/C+ testing with repeated user measurements
    - Recommendation systems: comparing algorithms for same user cohorts
    - Multi-task learning: comparing model performance across related tasks
    - Ensemble methods: comparing individual models within same data splits
    - Personalized medicine: comparing treatments within patient groups

    **Choose Friedman test over repeated measures ANOVA when:**
    - Non-normal repeated measures data
    - Ordinal outcome variables
    - Presence of outliers in repeated measurements
    - Small sample sizes where normality cannot be assumed
    - Robust analysis preferred

    **Choose Friedman over other tests when:**
    - 3+ related conditions (use Wilcoxon for 2 conditions)
    - Within-subjects or matched design
    - Non-parametric analysis required
    - Data cannot be meaningfully transformed to normality

    Parameters
    ----------
    effect_size : Tensor
        Effect size representing the variance of treatment effects relative
        to the error variance in the rank domain.
    n_subjects : Tensor
        Number of subjects (blocks) in the repeated measures design.
    n_treatments : Tensor
        Number of treatments (conditions) being compared.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    output : Tensor, shape=(...)
        Statistical power values in range [0, 1].

    Examples
    --------
    >>> effect_size = torch.tensor(0.4)
    >>> n_subjects = torch.tensor(20)
    >>> n_treatments = torch.tensor(3)
    >>> friedman_test_power(effect_size, n_subjects, n_treatments)
    tensor(0.8125)

    Notes
    -----
    The Friedman test statistic is:

    χ²_F = 12/(n*k*(k+1)) * Σ(R_j²) - 3*n*(k+1)

    where:
    - n = number of subjects (blocks)
    - k = number of treatments
    - R_j = sum of ranks for treatment j

    Under H₀, this follows approximately χ²(k-1).
    Under H₁, it follows approximately noncentral χ²(k-1, λ).

    The noncentrality parameter is approximated as:
    λ = 12 * n * Σ(τ_j²) / (k * (k+1))

    where τ_j represents the treatment effect for condition j.
    The effect_size parameter represents the standardized variance of these
    treatment effects.

    References
    ----------
    Hollander, M., Wolfe, D. A., & Chicken, E. (2013).
    Nonparametric statistical methods. John Wiley & Sons.
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    n_subjects = torch.atleast_1d(torch.as_tensor(n_subjects))
    n_treatments = torch.atleast_1d(torch.as_tensor(n_treatments))

    # Ensure floating point dtype
    dtype = torch.result_type(effect_size, n_subjects, n_treatments)
    if not dtype.is_floating_point:
        dtype = torch.float32
    effect_size = effect_size.to(dtype)
    n_subjects = n_subjects.to(dtype)
    n_treatments = n_treatments.to(dtype)

    # Validate inputs
    effect_size = torch.clamp(effect_size, min=0.0)
    n_subjects = torch.clamp(n_subjects, min=3.0)
    n_treatments = torch.clamp(n_treatments, min=3.0)

    # Degrees of freedom
    df = n_treatments - 1

    # Noncentrality parameter approximation
    # Based on the variance of treatment ranks under alternative hypothesis
    lambda_nc = 12 * n_subjects * effect_size / (n_treatments * (n_treatments + 1))

    # Critical chi-square value using normal approximation
    sqrt2 = math.sqrt(2.0)
    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
    chi2_critical = df + z_alpha * torch.sqrt(2 * df)

    # Noncentral chi-square approximation
    # Under H1: χ²_F ~ χ²(df, λ) ≈ N(df + λ, 2(df + 2λ))
    mean_nc_chi2 = df + lambda_nc
    var_nc_chi2 = 2 * (df + 2 * lambda_nc)
    std_nc_chi2 = torch.sqrt(torch.clamp(var_nc_chi2, min=1e-12))

    # Standardized test statistic
    z_score = (chi2_critical - mean_nc_chi2) / std_nc_chi2

    # Power = P(χ²_F > χ²_critical | H1) = P(Z > z_score)
    power = 0.5 * (1 - torch.erf(z_score / sqrt2))

    # Clamp to valid range
    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)
        return out
    return power
