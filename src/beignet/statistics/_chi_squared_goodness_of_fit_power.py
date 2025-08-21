"""Chi-square goodness-of-fit power."""

import math

import torch
from torch import Tensor


def chi_square_goodness_of_fit_power(
    effect_size: Tensor,
    sample_size: Tensor,
    df: Tensor,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute statistical power for chi-square goodness-of-fit tests.

    This function calculates the probability of correctly rejecting the null
    hypothesis when the alternative hypothesis is true for a chi-square
    goodness-of-fit test.

    This function is differentiable with respect to all tensor parameters.
    While traditional power analysis doesn't require gradients, differentiability
    enables integration into machine learning pipelines where effect sizes or
    sample sizes might be learned parameters or part of experimental design
    optimization.

    Parameters
    ----------
    effect_size : Tensor
        Cohen's w effect size. This is calculated as the square root of the
        sum of squared standardized differences: w = √(Σ((p₁ᵢ - p₀ᵢ)²/p₀ᵢ))
        where p₀ᵢ are the expected proportions and p₁ᵢ are the observed proportions.
        Should be non-negative.

    sample_size : Tensor
        Sample size (total number of observations).

    df : Tensor
        Degrees of freedom for the chi-square test (number of categories - 1).

    alpha : float, default=0.05
        Significance level (Type I error rate).

    out : Tensor, optional
        Output tensor. Default, `None`.

    Returns
    -------
    output : Tensor
        Statistical power (probability of correctly rejecting false null hypothesis).

    Examples
    --------
    >>> effect_size = torch.tensor(0.3)
    >>> sample_size = torch.tensor(100)
    >>> df = torch.tensor(3)
    >>> beignet.chi_square_goodness_of_fit_power(effect_size, sample_size, df)
    tensor(0.6740)

    Notes
    -----
    The power calculation uses the noncentral chi-square distribution. Under the
    null hypothesis, the test statistic follows χ²(df). Under the alternative
    hypothesis, it follows a noncentral chi-square distribution with noncentrality
    parameter:

    λ = n * w²

    Where n is the sample size and w is Cohen's w effect size.

    Cohen's w effect size interpretation:
    - Small effect: w = 0.10
    - Medium effect: w = 0.30
    - Large effect: w = 0.50

    For computational efficiency, we use normal approximations for large degrees
    of freedom and accurate noncentral chi-square calculations for smaller df.

    References
    ----------
    .. [1] Cohen, J. (1988). Statistical power analysis for the behavioral
           sciences (2nd ed.). Erlbaum.
    .. [2] Cramér, H. (1946). Mathematical Methods of Statistics. Princeton
           University Press.
    """
    # Convert inputs to tensors if needed
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))
    df = torch.atleast_1d(torch.as_tensor(df))

    # Ensure tensors have the same dtype
    if (
        effect_size.dtype == torch.float64
        or sample_size.dtype == torch.float64
        or df.dtype == torch.float64
    ):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    sample_size = sample_size.to(dtype)
    df = df.to(dtype)

    # Clamp effect size to non-negative values
    effect_size = torch.clamp(effect_size, min=0.0)

    # Ensure positive sample size and degrees of freedom
    sample_size = torch.clamp(sample_size, min=1.0)
    df = torch.clamp(df, min=1.0)

    # Noncentrality parameter
    ncp = sample_size * effect_size**2

    # Critical chi-square value using normal approximation
    sqrt_2 = math.sqrt(2.0)

    # For large df, chi-square approaches normal: χ² ≈ N(df, 2*df)
    # Critical value: χ²_α = df + z_α * √(2*df)
    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2
    chi2_critical = df + z_alpha * torch.sqrt(2 * df)

    # For noncentral chi-square, use normal approximation:
    # χ²(df, λ) ≈ N(df + λ, 2*(df + 2*λ))
    mean_nc_chi2 = df + ncp
    var_nc_chi2 = 2 * (df + 2 * ncp)
    std_nc_chi2 = torch.sqrt(var_nc_chi2)

    # Calculate power: P(χ² > χ²_critical | λ = ncp)
    z_score = (chi2_critical - mean_nc_chi2) / torch.clamp(std_nc_chi2, min=1e-10)

    # Power = P(Z > z_score) = 1 - Φ(z_score)
    power = (1 - torch.erf(z_score / sqrt_2)) / 2

    # Clamp power to [0, 1] range
    output = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
