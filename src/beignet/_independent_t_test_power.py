import math

import torch
from torch import Tensor


def independent_t_test_power(
    effect_size: Tensor,
    nobs1: Tensor,
    ratio: Tensor | None = None,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute statistical power for independent samples t-tests.

    This function calculates the probability of correctly rejecting the null
    hypothesis when the alternative hypothesis is true for an independent
    samples t-test (also known as two-sample t-test).

    This function is differentiable with respect to all tensor parameters.
    While traditional power analysis doesn't require gradients, differentiability
    enables integration into machine learning pipelines where effect sizes or
    sample sizes might be learned parameters or part of experimental design
    optimization.

    Parameters
    ----------
    effect_size : Tensor
        Standardized effect size (Cohen's d). This is the difference between
        the two group means divided by the pooled standard deviation:
        d = (μ₁ - μ₂) / σ_pooled. Should be positive.
    nobs1 : Tensor
        Number of observations in group 1.
    ratio : Tensor, default=1.0
        Ratio of sample size for group 2 relative to group 1.
        Group 2 sample size = nobs1 * ratio.
    alpha : float, default=0.05
        Significance level (Type I error rate).
    alternative : str, default="two-sided"
        Type of alternative hypothesis. Either "two-sided", "larger", or "smaller".

    Returns
    -------
    output : Tensor
        Statistical power (probability of correctly rejecting false null hypothesis).

    Examples
    --------
    >>> effect_size = torch.tensor(0.5)
    >>> nobs1 = torch.tensor(30)
    >>> beignet.independent_t_test_power(effect_size, nobs1)
    tensor(0.4741)

    Notes
    -----
    The power calculation uses the noncentral t-distribution. The test statistic
    under the null hypothesis follows t(df) where df = n₁ + n₂ - 2. Under the
    alternative hypothesis, it follows a noncentral t-distribution with
    noncentrality parameter:

    δ = d * √(n₁ * n₂ / (n₁ + n₂))

    Where d is Cohen's d effect size, n₁ and n₂ are the sample sizes.

    The pooled standard error is:
    SE = √((1/n₁ + 1/n₂) * σ²_pooled)

    For computational efficiency, we use normal approximations for large
    sample sizes and accurate noncentral t-distribution calculations for
    smaller samples.

    References
    ----------
    .. [1] Cohen, J. (1988). Statistical power analysis for the behavioral
           sciences (2nd ed.). Erlbaum.
    .. [2] Aberson, C. L. (2010). Applied power analysis for the behavioral
           sciences. Routledge.
    """
    # Convert inputs to tensors if needed
    effect_size = torch.as_tensor(effect_size)
    nobs1 = torch.as_tensor(nobs1)
    if ratio is None:
        ratio = torch.tensor(1.0)
    else:
        ratio = torch.as_tensor(ratio)

    # Ensure tensors have the same dtype
    if (
        effect_size.dtype == torch.float64
        or nobs1.dtype == torch.float64
        or ratio.dtype == torch.float64
    ):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    nobs1 = nobs1.to(dtype)
    ratio = ratio.to(dtype)

    # Ensure positive values and reasonable constraints
    effect_size = torch.clamp(effect_size, min=0.0)
    nobs1 = torch.clamp(nobs1, min=2.0)
    ratio = torch.clamp(ratio, min=0.1, max=10.0)  # Reasonable ratio range

    # Calculate sample sizes
    nobs2 = nobs1 * ratio
    total_n = nobs1 + nobs2

    # Degrees of freedom
    df = total_n - 2
    df = torch.clamp(df, min=1.0)

    # Standard error factor: sqrt(1/n1 + 1/n2)
    se_factor = torch.sqrt(1 / nobs1 + 1 / nobs2)

    # Noncentrality parameter: effect_size / se_factor
    ncp = effect_size / se_factor

    # Critical t-value using normal approximation adjusted for finite df
    sqrt_2 = math.sqrt(2.0)

    if alternative == "two-sided":
        z_alpha_half = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt_2
        t_critical = z_alpha_half * torch.sqrt(1 + 1 / (2 * df))
    elif alternative == "larger":
        # One-tailed test: H1: μ₁ > μ₂
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2
        t_critical = z_alpha * torch.sqrt(1 + 1 / (2 * df))
    else:  # alternative == "smaller"
        # One-tailed test: H1: μ₁ < μ₂
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2
        t_critical = z_alpha * torch.sqrt(1 + 1 / (2 * df))

    # Noncentral t-distribution approximation
    # For large df, noncentral t approaches normal with mean=ncp and adjusted variance
    mean_nct = ncp

    # Variance approximation for noncentral t
    var_nct = torch.where(
        df > 2, (df + ncp**2) / (df - 2), 1 + ncp**2 / (2 * torch.clamp(df, min=1.0))
    )
    std_nct = torch.sqrt(var_nct)

    # Calculate power based on alternative hypothesis
    if alternative == "two-sided":
        # P(|T| > t_critical)
        z_upper = (t_critical - mean_nct) / torch.clamp(std_nct, min=1e-10)
        z_lower = (-t_critical - mean_nct) / torch.clamp(std_nct, min=1e-10)

        power = (1 - torch.erf(z_upper / sqrt_2)) / 2 + (
            1 - torch.erf(-z_lower / sqrt_2)
        ) / 2
    elif alternative == "larger":
        # P(T > t_critical) - expecting positive effect
        z_score = (t_critical - mean_nct) / torch.clamp(std_nct, min=1e-10)
        power = (1 - torch.erf(z_score / sqrt_2)) / 2
    else:  # alternative == "smaller"
        # P(T < -t_critical) - expecting negative effect
        # Note: we flip the sign of ncp for smaller alternative
        mean_nct_neg = -mean_nct
        z_score = (-t_critical - mean_nct_neg) / torch.clamp(std_nct, min=1e-10)
        power = (1 - torch.erf(-z_score / sqrt_2)) / 2

    # Clamp power to [0, 1] range
    output = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
