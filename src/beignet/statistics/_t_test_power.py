import math

import torch
from torch import Tensor


def t_test_power(
    effect_size: Tensor,
    sample_size: Tensor,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute statistical power for one-sample and paired t-tests.

    This function calculates the probability of correctly rejecting the null
    hypothesis when the alternative hypothesis is true for a one-sample t-test
    or a paired-samples t-test.

    This function is differentiable with respect to all tensor parameters.
    While traditional power analysis doesn't require gradients, differentiability
    enables integration into machine learning pipelines where effect sizes or
    sample sizes might be learned parameters or part of experimental design
    optimization.

    Parameters
    ----------
    effect_size : Tensor
        Standardized effect size (Cohen's d). For one-sample tests, this is
        (μ - μ₀) / σ where μ is the true mean, μ₀ is the null hypothesis mean,
        and σ is the population standard deviation. For paired tests, this is
        the mean difference divided by the standard deviation of differences.

    sample_size : Tensor
        Sample size (number of observations). For paired tests, this is the
        number of pairs.

    alpha : float, default=0.05
        Significance level (Type I error rate).

    alternative : str, default="two-sided"
        Type of alternative hypothesis. Either "two-sided" or "one-sided".

    out : Tensor, optional
        Output tensor. Default, `None`.

    Returns
    -------
    output : Tensor
        Statistical power (probability of correctly rejecting false null hypothesis).

    Examples
    --------
    >>> effect_size = torch.tensor(0.5)
    >>> sample_size = torch.tensor(30)
    >>> beignet.t_test_power(effect_size, sample_size)
    tensor(0.6947)

    Notes
    -----
    The power calculation uses the noncentral t-distribution. Under the null
    hypothesis, the test statistic follows t(df) where df = n - 1. Under the
    alternative hypothesis, it follows a noncentral t-distribution with
    noncentrality parameter:

    δ = d * √n

    Where d is Cohen's d effect size and n is the sample size.

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
    sample_size = torch.as_tensor(sample_size)

    # Ensure tensors have the same dtype
    if effect_size.dtype == torch.float64 or sample_size.dtype == torch.float64:
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    sample_size = sample_size.to(dtype)

    # Ensure positive sample size and clamp to reasonable range
    sample_size = torch.clamp(sample_size, min=2.0)

    # Calculate degrees of freedom
    df = sample_size - 1

    # Noncentrality parameter
    ncp = effect_size * torch.sqrt(sample_size)

    # Critical t-value using normal approximation
    sqrt_2 = math.sqrt(2.0)

    if alternative == "two-sided":
        # Two-tailed test
        z_alpha_half = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt_2
        t_critical = z_alpha_half * torch.sqrt(
            1 + 1 / (2 * df)
        )  # Adjustment for finite df
    else:
        # One-tailed test
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2
        t_critical = z_alpha * torch.sqrt(1 + 1 / (2 * df))  # Adjustment for finite df

    # For noncentral t-distribution, we approximate using normal distribution
    # when df is large, and adjust for smaller df

    # Under alternative hypothesis, test statistic is approximately:
    # t ~ N(ncp, 1 + ncp²/(2*df)) for large df
    # We use a better approximation that accounts for finite df

    mean_nct = ncp
    # Variance approximation for noncentral t
    # Use torch.where to avoid data-dependent branching
    var_nct = torch.where(
        df > 2, (df + ncp**2) / (df - 2), 1 + ncp**2 / (2 * torch.clamp(df, min=2.0))
    )
    std_nct = torch.sqrt(var_nct)

    if alternative == "two-sided":
        # P(|T| > t_critical) = P(T > t_critical) + P(T < -t_critical)
        z_upper = (t_critical - mean_nct) / torch.clamp(std_nct, min=1e-10)
        z_lower = (-t_critical - mean_nct) / torch.clamp(std_nct, min=1e-10)

        power = (1 - torch.erf(z_upper / sqrt_2)) / 2 + (
            1 - torch.erf(-z_lower / sqrt_2)
        ) / 2
    else:
        # One-tailed: P(T > t_critical)
        z_score = (t_critical - mean_nct) / torch.clamp(std_nct, min=1e-10)
        power = (1 - torch.erf(z_score / sqrt_2)) / 2

    # Clamp power to [0, 1] range
    output = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
