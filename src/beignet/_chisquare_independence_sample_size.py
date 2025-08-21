import math

import torch
from torch import Tensor


def chisquare_independence_sample_size(
    effect_size: Tensor,
    rows: Tensor,
    cols: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute required sample size for chi-square independence tests.

    Given the effect size, contingency table dimensions, desired power, and
    significance level, this function calculates the minimum sample size needed
    to achieve the specified power for a chi-square test of independence.

    This function is differentiable with respect to effect_size, rows, and cols
    parameters. While traditional sample size calculations don't require gradients,
    differentiability enables integration into machine learning pipelines where
    effect sizes might be learned parameters or part of experimental design
    optimization.

    Parameters
    ----------
    effect_size : Tensor
        Cohen's w effect size. For independence tests, this measures the strength
        of association between two categorical variables. It can be calculated as
        w = √(χ²/n) where χ² is the chi-square statistic and n is the sample size.
        Should be positive.
    rows : Tensor
        Number of rows in the contingency table (categories of first variable).
    cols : Tensor
        Number of columns in the contingency table (categories of second variable).
    power : float, default=0.8
        Desired statistical power (probability of correctly rejecting false null).
    alpha : float, default=0.05
        Significance level (Type I error rate).

    Returns
    -------
    output : Tensor
        Required sample size (rounded up to nearest integer).

    Examples
    --------
    >>> effect_size = torch.tensor(0.3)
    >>> rows = torch.tensor(3)
    >>> cols = torch.tensor(3)
    >>> beignet.chisquare_independence_sample_size(effect_size, rows, cols, power=0.8)
    tensor(124)

    Notes
    -----
    The sample size calculation is based on the noncentral chi-square distribution.
    For a chi-square independence test with effect size w and sample size n,
    the noncentrality parameter is:

    λ = n * w²

    The degrees of freedom are: df = (rows - 1) × (cols - 1)

    The calculation uses an iterative approach to find the sample size that
    achieves the desired power, starting from an initial normal approximation:

    n ≈ ((z_α + z_β) / w)²

    Where z_α and z_β are the critical values for the given α and β = 1 - power.

    For computational efficiency, we use analytical approximations where possible,
    falling back to iterative refinement when needed.

    References
    ----------
    .. [1] Cohen, J. (1988). Statistical power analysis for the behavioral
           sciences (2nd ed.). Erlbaum.
    .. [2] Cramér, H. (1946). Mathematical Methods of Statistics. Princeton
           University Press.
    """
    # Convert inputs to tensors if needed
    effect_size = torch.as_tensor(effect_size)
    rows = torch.as_tensor(rows)
    cols = torch.as_tensor(cols)

    # Ensure tensors have the same dtype
    if (
        effect_size.dtype == torch.float64
        or rows.dtype == torch.float64
        or cols.dtype == torch.float64
    ):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    rows = rows.to(dtype)
    cols = cols.to(dtype)

    # Clamp effect size to positive values and ensure at least 2 categories
    effect_size = torch.clamp(effect_size, min=1e-6)
    rows = torch.clamp(rows, min=2.0)
    cols = torch.clamp(cols, min=2.0)

    # Calculate degrees of freedom for independence test
    df = (rows - 1) * (cols - 1)

    # Standard normal quantiles using erfinv
    sqrt_2 = math.sqrt(2.0)
    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2
    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt_2

    # Initial normal approximation for chi-square test
    # For large sample sizes, the approximation is: n ≈ ((z_α + z_β) / w)²
    n_initial = ((z_alpha + z_beta) / effect_size) ** 2

    # Ensure minimum sample size (rule of thumb: at least 5 expected in each cell)
    min_sample_size = 5.0 * rows * cols
    n_initial = torch.clamp(n_initial, min=min_sample_size)

    # Iterative refinement to account for finite df effects
    n_current = n_initial

    for _ in range(5):  # Usually converges in 3-4 iterations
        # Current noncentrality parameter
        ncp_current = n_current * effect_size**2

        # Critical chi-square value using normal approximation
        # χ²_α = df + z_α * √(2*df)
        chi2_critical = df + z_alpha * torch.sqrt(2 * df)

        # For noncentral chi-square, use normal approximation
        # χ²(df, λ) ≈ N(df + λ, 2*(df + 2*λ))
        mean_nc_chi2 = df + ncp_current
        var_nc_chi2 = 2 * (df + 2 * ncp_current)
        std_nc_chi2 = torch.sqrt(var_nc_chi2)

        # Calculate current power
        z_score = (chi2_critical - mean_nc_chi2) / torch.clamp(std_nc_chi2, min=1e-10)
        power_current = (1 - torch.erf(z_score / sqrt_2)) / 2

        # Clamp power to valid range
        power_current = torch.clamp(power_current, 0.01, 0.99)

        # Calculate power difference
        power_diff = power - power_current

        # Newton-Raphson style adjustment
        # Approximate derivative: d(power)/d(n) ≈ d(power)/d(λ) * w²
        adjustment = (
            power_diff
            * n_current
            / (2 * torch.clamp(power_current * (1 - power_current), min=0.01))
        )
        n_current = n_current + adjustment

        # Ensure minimum constraints
        n_current = torch.clamp(n_current, min=min_sample_size)
        n_current = torch.clamp(n_current, max=1000000.0)

    # Round up to nearest integer
    output = torch.ceil(n_current)

    # Final check: ensure we meet minimum sample size requirements
    output = torch.clamp(output, min=min_sample_size)

    if out is not None:
        out.copy_(output)
        return out

    return output
