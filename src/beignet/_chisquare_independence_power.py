import math

import torch
from torch import Tensor


def chisquare_independence_power(
    effect_size: Tensor,
    sample_size: Tensor,
    rows: Tensor,
    cols: Tensor,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute statistical power for chi-square independence tests.

    This function calculates the probability of correctly rejecting the null
    hypothesis when the alternative hypothesis is true for a chi-square test
    of independence between two categorical variables.

    This function is differentiable with respect to all tensor parameters.
    While traditional power analysis doesn't require gradients, differentiability
    enables integration into machine learning pipelines where effect sizes or
    sample sizes might be learned parameters or part of experimental design
    optimization.

    Parameters
    ----------
    effect_size : Tensor
        Cohen's w effect size. For independence tests, this measures the strength
        of association between two categorical variables. It can be calculated as
        w = √(χ²/n) where χ² is the chi-square statistic and n is the sample size.
        Should be non-negative.
    sample_size : Tensor
        Sample size (total number of observations).
    rows : Tensor
        Number of rows in the contingency table (categories of first variable).
    cols : Tensor
        Number of columns in the contingency table (categories of second variable).
    alpha : float, default=0.05
        Significance level (Type I error rate).

    Returns
    -------
    output : Tensor
        Statistical power (probability of correctly rejecting false null hypothesis).

    Examples
    --------
    >>> effect_size = torch.tensor(0.3)
    >>> sample_size = torch.tensor(100)
    >>> rows = torch.tensor(3)
    >>> cols = torch.tensor(3)
    >>> beignet.chisquare_independence_power(effect_size, sample_size, rows, cols)
    tensor(0.7985)

    Notes
    -----
    The power calculation uses the noncentral chi-square distribution. Under the
    null hypothesis of independence, the test statistic follows χ²(df) where
    df = (rows - 1) × (cols - 1). Under the alternative hypothesis, it follows
    a noncentral chi-square distribution with noncentrality parameter:

    λ = n * w²

    Where n is the sample size and w is Cohen's w effect size.

    Cohen's w effect size interpretation:
    - Small effect: w = 0.10
    - Medium effect: w = 0.30
    - Large effect: w = 0.50

    The degrees of freedom for independence tests differ from goodness-of-fit
    tests: df = (r-1)(c-1) where r is the number of rows and c is the number
    of columns in the contingency table.

    References
    ----------
    .. [1] Cohen, J. (1988). Statistical power analysis for the behavioral
           sciences (2nd ed.). Erlbaum.
    .. [2] Cramér, H. (1946). Mathematical Methods of Statistics. Princeton
           University Press.
    """
    # Convert inputs to tensors if needed
    effect_size = torch.as_tensor(effect_size)
    sample_size = torch.as_tensor(sample_size)
    rows = torch.as_tensor(rows)
    cols = torch.as_tensor(cols)

    # Ensure tensors have the same dtype
    if (
        effect_size.dtype == torch.float64
        or sample_size.dtype == torch.float64
        or rows.dtype == torch.float64
        or cols.dtype == torch.float64
    ):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    sample_size = sample_size.to(dtype)
    rows = rows.to(dtype)
    cols = cols.to(dtype)

    # Clamp effect size to non-negative values
    effect_size = torch.clamp(effect_size, min=0.0)

    # Ensure positive sample size and at least 2 categories for each variable
    sample_size = torch.clamp(sample_size, min=1.0)
    rows = torch.clamp(rows, min=2.0)
    cols = torch.clamp(cols, min=2.0)

    # Calculate degrees of freedom for independence test
    df = (rows - 1) * (cols - 1)

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
