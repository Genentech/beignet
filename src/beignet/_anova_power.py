import math

import torch
from torch import Tensor


def anova_power(
    effect_size: Tensor,
    sample_size: Tensor,
    k: Tensor,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute statistical power for one-way ANOVA F-tests.

    Given Cohen's f effect size, sample size, and number of groups,
    this function calculates the probability of correctly rejecting
    the false null hypothesis of equal group means (statistical power).

    This function is differentiable with respect to all tensor parameters.
    While traditional power analysis doesn't require gradients, differentiability
    enables integration into machine learning pipelines where effect sizes or
    sample sizes might be learned parameters or part of experimental design
    optimization.

    Parameters
    ----------
    effect_size : Tensor
        Cohen's f effect size. Should be non-negative.
    sample_size : Tensor
        Total sample size across all groups.
    k : Tensor
        Number of groups in the ANOVA.
    alpha : float, default=0.05
        Significance level (Type I error rate).

    Returns
    -------
    output : Tensor
        Statistical power (probability of correctly rejecting false null hypothesis).

    Examples
    --------
    >>> effect_size = torch.tensor(0.25)
    >>> sample_size = torch.tensor(120)
    >>> k = torch.tensor(3)
    >>> beignet.anova_power(effect_size, sample_size, k)
    tensor(0.7061)

    Notes
    -----
    The power calculation uses the noncentral F-distribution. The test statistic
    under the null hypothesis follows F(k-1, N-k), where N is the total sample
    size and k is the number of groups.

    Under the alternative hypothesis, the test statistic follows a noncentral
    F-distribution with noncentrality parameter:

    λ = N * f²

    Where f is Cohen's f effect size.

    The degrees of freedom are:
    - df₁ = k - 1 (between groups)
    - df₂ = N - k (within groups)

    For computational efficiency, we use the approximation that for large df₂,
    the noncentral F-distribution can be approximated using the noncentral
    chi-square distribution.

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
    sample_size = torch.as_tensor(sample_size)
    k = torch.as_tensor(k)

    # Ensure all tensors have the same dtype
    if (
        effect_size.dtype == torch.float64
        or sample_size.dtype == torch.float64
        or k.dtype == torch.float64
    ):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    sample_size = sample_size.to(dtype)
    k = k.to(dtype)

    # Clamp effect size to non-negative values
    effect_size = torch.clamp(effect_size, min=0.0)

    # Calculate degrees of freedom
    df1 = k - 1  # Between groups
    df2 = sample_size - k  # Within groups (error)

    # Ensure we have positive degrees of freedom
    df1 = torch.clamp(df1, min=1.0)
    df2 = torch.clamp(df2, min=1.0)

    # Critical F-value for given alpha
    # For large df2, F_{alpha,df1,df2} ≈ χ²_{alpha,df1} / df1
    # We'll use a gamma distribution approximation for the F-distribution critical value

    # Use the relationship: if X ~ F(df1,df2), then (df1*X) ~ scaled version of chi-square
    # For simplicity, we'll use the chi-square approximation when df2 is large

    # Calculate critical chi-square value using erfinv
    sqrt_2 = math.sqrt(2.0)

    # Use normal approximation for chi-square critical value
    # χ² ≈ N(df1, 2*df1) for large df1, but works reasonably for smaller df1 too
    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2
    chi2_critical = df1 + z_alpha * torch.sqrt(2 * df1)

    # Convert back to F critical value
    f_critical = chi2_critical / df1

    # Noncentrality parameter
    lambda_nc = sample_size * effect_size**2

    # Under the alternative hypothesis, the F-statistic follows a noncentral F-distribution
    # For power calculation, we need P(F > f_critical | λ = lambda_nc)

    # Use approximation for noncentral F-distribution
    # The noncentral F can be approximated as: (χ²(df1, λ) / df1) / (χ²(df2) / df2)

    # For large df2, the denominator approaches 1, so we have χ²(df1, λ) / df1
    # The noncentral chi-square with noncentrality λ can be approximated as
    # normal with mean (df1 + λ) and variance 2*(df1 + 2*λ)

    mean_nc_chi2 = df1 + lambda_nc
    var_nc_chi2 = 2 * (df1 + 2 * lambda_nc)

    # Convert to F-statistic distribution parameters
    mean_f = mean_nc_chi2 / df1
    var_f = var_nc_chi2 / (df1**2)

    # Approximate adjustment for finite df2
    # F-ratio has additional variability from denominator
    # Use a smooth adjustment function instead of hard threshold
    adjustment_factor = (df2 + 2) / torch.clamp(df2, min=1.0)
    var_f = var_f * adjustment_factor

    # Calculate power using normal approximation
    std_f = torch.sqrt(var_f)
    z_score = (f_critical - mean_f) / torch.clamp(std_f, min=1e-10)

    # Power = P(F > f_critical) = P(Z > z_score) = 1 - Φ(z_score)
    power = (1 - torch.erf(z_score / sqrt_2)) / 2

    # Clamp power to [0, 1] range
    output = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
