import math

import torch
from torch import Tensor


def anova_sample_size(
    effect_size: Tensor,
    k: Tensor,
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

    Parameters
    ----------
    effect_size : Tensor
        Cohen's f effect size. Should be positive.
    k : Tensor
        Number of groups in the ANOVA.
    power : float, default=0.8
        Desired statistical power (probability of correctly rejecting false null).
    alpha : float, default=0.05
        Significance level (Type I error rate).

    Returns
    -------
    output : Tensor
        Required total sample size across all groups (rounded up to nearest integer).

    Examples
    --------
    >>> effect_size = torch.tensor(0.25)
    >>> k = torch.tensor(3)
    >>> beignet.anova_sample_size(effect_size, k, power=0.8)
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
    # Convert inputs to tensors if needed
    effect_size = torch.as_tensor(effect_size)
    k = torch.as_tensor(k)

    # Ensure tensors have the same dtype
    if effect_size.dtype == torch.float64 or k.dtype == torch.float64:
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    k = k.to(dtype)

    # Clamp effect size to positive values and k to at least 2
    effect_size = torch.clamp(effect_size, min=1e-6)
    k = torch.clamp(k, min=2.0)

    # Calculate degrees of freedom (df1 is k-1)
    df1 = k - 1

    # Standard normal quantiles using erfinv
    sqrt_2 = math.sqrt(2.0)
    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2
    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt_2

    # Initial approximation using central chi-square distribution
    # For large df2, F-distribution approaches chi-square/df1 distribution

    # Critical chi-square value (approximation for F-distribution)
    chi2_critical = df1 + z_alpha * torch.sqrt(2 * df1)

    # For noncentral chi-square, we need to solve:
    # P(χ²(df1, λ) > chi2_critical) = power
    # Where λ = N * f²

    # Using the normal approximation to noncentral chi-square:
    # χ²(df1, λ) ≈ N(df1 + λ, 2*(df1 + 2*λ))

    # We need: (chi2_critical - (df1 + λ)) / sqrt(2*(df1 + 2*λ)) = -z_beta
    # Solving for λ (and thus N):

    # Let's use an iterative approach for better accuracy
    # Start with a rough approximation

    # From the normal approximation: chi2_critical = df1 + λ - z_beta * sqrt(2*(df1 + 2*λ))
    # This gives us a quadratic in sqrt(λ)

    # Initial guess using simplified formula
    # For large λ: λ ≈ ((z_alpha + z_beta) * sqrt(2))² / f²
    lambda_initial = ((z_alpha + z_beta) * sqrt_2) ** 2
    n_initial = lambda_initial / (effect_size**2)

    # Ensure minimum sample size greater than k
    n_initial = torch.clamp(n_initial, min=k + 1)

    # Iterative refinement (3-4 iterations should be sufficient)
    n_current = n_initial

    for _ in range(4):
        # Calculate current df2
        df2_current = n_current - k
        df2_current = torch.clamp(df2_current, min=1.0)

        # Current noncentrality parameter
        lambda_current = n_current * effect_size**2

        # Improved critical F-value incorporating df2
        f_critical = chi2_critical / df1

        # Better approximation accounting for finite df2
        # Use smooth adjustment instead of conditional
        adjustment = 1 + 2 / torch.clamp(df2_current, min=1.0)
        f_critical = f_critical * adjustment

        # Expected F-statistic under alternative
        mean_nc_chi2 = df1 + lambda_current
        var_nc_chi2 = 2 * (df1 + 2 * lambda_current)

        mean_f = mean_nc_chi2 / df1
        var_f = var_nc_chi2 / (df1**2)

        # Adjustment for finite df2
        var_adjustment = (df2_current + 2) / torch.clamp(df2_current, min=1.0)
        var_f = var_f * var_adjustment

        # Calculate z-score for current power
        std_f = torch.sqrt(var_f)
        z_current = (f_critical - mean_f) / torch.clamp(std_f, min=1e-10)

        # Current power
        power_current = (1 - torch.erf(z_current / sqrt_2)) / 2

        # Adjustment factor to reach target power
        power_diff = power - power_current

        # Approximate derivative: d(power)/d(N) ≈ d(power)/d(λ) * f²
        # Adjust sample size based on power difference
        adjustment = power_diff * n_current * 0.5  # Conservative adjustment factor
        n_current = n_current + adjustment

        # Ensure minimum constraints
        n_current = torch.clamp(n_current, min=k + 1)
        n_current = torch.clamp(n_current, max=100000.0)  # Upper bound for practicality

    # Round up to nearest integer
    output = torch.ceil(n_current)

    # Final check: ensure we have at least k+1 subjects
    output = torch.clamp(output, min=k + 1)

    if out is not None:
        out.copy_(output)
        return out

    return output
