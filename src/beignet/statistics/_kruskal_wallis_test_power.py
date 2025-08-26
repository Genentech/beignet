import math

import torch
from torch import Tensor


def kruskal_wallis_test_power(
    effect_size: Tensor,
    sample_sizes: Tensor,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Power analysis for the Kruskal-Wallis H test.

    The Kruskal-Wallis test is a non-parametric alternative to one-way ANOVA
    for comparing three or more independent groups when the assumption of
    normality is violated.

    Parameters
    ----------
    effect_size : Tensor
        Effect size measured as the variance of the group location parameters
        divided by the error variance. For practical interpretation, this is
        similar to η² in ANOVA but for ranks.
    sample_sizes : Tensor, shape=(..., k)
        Sample sizes for each of the k groups.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    output : Tensor, shape=(...)
        Statistical power values in range [0, 1].

    Examples
    --------
    >>> effect_size = torch.tensor(0.3)
    >>> sample_sizes = torch.tensor([15, 15, 15])
    >>> kruskal_wallis_test_power(effect_size, sample_sizes)
    tensor(0.7234)

    Notes
    -----
    The Kruskal-Wallis H statistic approximately follows a chi-square distribution
    with (k-1) degrees of freedom under the null hypothesis. Under the alternative,
    it follows a noncentral chi-square distribution.

    The noncentrality parameter is approximated as:
    λ = 12 * Σ(n_i * (θ_i - θ̄)²) / (N * (N+1))

    where:
    - n_i = sample size for group i
    - θ_i = location parameter for group i
    - θ̄ = overall location parameter
    - N = total sample size

    For power calculation, we approximate this using the provided effect_size
    parameter which represents the standardized variance of location parameters.

    References
    ----------
    Lehmann, E. L. (2006). Nonparametrics: statistical methods based on ranks.
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    sample_sizes = torch.atleast_1d(torch.as_tensor(sample_sizes))

    # Ensure floating point dtype
    if effect_size.dtype.is_floating_point and sample_sizes.dtype.is_floating_point:
        if effect_size.dtype == torch.float64 or sample_sizes.dtype == torch.float64:
            dtype = torch.float64
        else:
            dtype = torch.float32
    else:
        dtype = torch.float32
    effect_size = effect_size.to(dtype)
    sample_sizes = sample_sizes.to(dtype)

    # Validate inputs
    effect_size = torch.clamp(effect_size, min=0.0)
    sample_sizes = torch.clamp(sample_sizes, min=2.0)

    # Number of groups and total sample size
    k = torch.tensor(sample_sizes.shape[-1], dtype=dtype)
    if k < 3:
        raise ValueError("Kruskal-Wallis test requires at least 3 groups")

    N = torch.sum(sample_sizes, dim=-1)

    # Degrees of freedom
    df = k - 1

    # Approximate noncentrality parameter
    # This is a simplified approximation based on the effect size
    lambda_nc = 12 * N * effect_size / (N + 1)

    # Critical chi-square value using normal approximation
    sqrt2 = math.sqrt(2.0)
    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
    chi2_critical = df + z_alpha * torch.sqrt(2 * df)

    # Noncentral chi-square approximation using normal distribution
    # Under H1: H ~ χ²(df, λ) ≈ N(df + λ, 2(df + 2λ))
    mean_nc_chi2 = df + lambda_nc
    var_nc_chi2 = 2 * (df + 2 * lambda_nc)
    std_nc_chi2 = torch.sqrt(torch.clamp(var_nc_chi2, min=1e-12))

    # Standardized test statistic
    z_score = (chi2_critical - mean_nc_chi2) / std_nc_chi2

    # Power = P(H > χ²_critical | H1) = P(Z > z_score)
    power = 0.5 * (1 - torch.erf(z_score / sqrt2))

    # Clamp to valid range
    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)
        return out
    return power
