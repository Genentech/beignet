import math

import torch
from torch import Tensor


def proportion_two_sample_power(
    p1: Tensor,
    p2: Tensor,
    n1: Tensor,
    n2: Tensor | None = None,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute statistical power for two-sample proportion tests.

    Given proportions for two groups and their sample sizes, this function
    calculates the probability of correctly detecting a difference between
    the proportions (statistical power).

    This function is differentiable with respect to all tensor parameters.
    While traditional power analysis doesn't require gradients, differentiability
    enables integration into machine learning pipelines where proportions or
    sample sizes might be learned parameters or part of experimental design
    optimization.

    Parameters
    ----------
    p1 : Tensor
        Proportion in group 1 (between 0 and 1).
    p2 : Tensor
        Proportion in group 2 (between 0 and 1).
    n1 : Tensor
        Sample size for group 1.
    n2 : Tensor, optional
        Sample size for group 2. If None, assumes equal sample sizes (n2 = n1).
    alpha : float, default=0.05
        Significance level (Type I error rate).
    alternative : str, default="two-sided"
        Type of alternative hypothesis. Options are "two-sided", "greater", or "less".

    Returns
    -------
    output : Tensor
        Statistical power (probability of correctly detecting the difference).

    Examples
    --------
    >>> p1 = torch.tensor(0.5)
    >>> p2 = torch.tensor(0.6)
    >>> n1 = torch.tensor(100)
    >>> n2 = torch.tensor(100)
    >>> beignet.proportion_two_sample_power(p1, p2, n1, n2)
    tensor(0.5592)

    Notes
    -----
    The test statistic follows a normal distribution under the null hypothesis
    of equal proportions. The test statistic is:

    Z = (p̂1 - p̂2) / sqrt(p̂*(1-p̂)*(1/n1 + 1/n2))

    Where p̂ is the pooled proportion estimate.

    Under the alternative hypothesis, the test statistic has mean:
    μ = (p1 - p2) / sqrt(p̂*(1-p̂)*(1/n1 + 1/n2))

    References
    ----------
    .. [1] Fleiss, J. L., Levin, B., & Paik, M. C. (2003). Statistical methods
           for rates and proportions. John Wiley & Sons.
    .. [2] Agresti, A. (2013). Categorical data analysis. John Wiley & Sons.
    """
    # Convert inputs to tensors if needed
    p1 = torch.as_tensor(p1)
    p2 = torch.as_tensor(p2)
    n1 = torch.as_tensor(n1)

    if n2 is None:
        n2 = n1
    else:
        n2 = torch.as_tensor(n2)

    # Ensure all tensors have the same dtype
    if (
        p1.dtype == torch.float64
        or p2.dtype == torch.float64
        or n1.dtype == torch.float64
        or n2.dtype == torch.float64
    ):
        dtype = torch.float64
    else:
        dtype = torch.float32

    p1 = p1.to(dtype)
    p2 = p2.to(dtype)
    n1 = n1.to(dtype)
    n2 = n2.to(dtype)

    # Clamp proportions to valid range (0, 1)
    epsilon = 1e-8
    p1 = torch.clamp(p1, epsilon, 1 - epsilon)
    p2 = torch.clamp(p2, epsilon, 1 - epsilon)

    # Pooled proportion under null hypothesis (p1 = p2)
    p_pooled = (n1 * p1 + n2 * p2) / (n1 + n2)
    p_pooled = torch.clamp(p_pooled, epsilon, 1 - epsilon)

    # Standard error under null hypothesis (pooled variance)
    se_null = torch.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))

    # Standard error under alternative hypothesis (separate variances)
    se_alt = torch.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)

    # Effect size (standardized difference)
    effect = (p1 - p2) / se_null

    # Standard normal quantiles using erfinv
    sqrt_2 = math.sqrt(2.0)

    # Effect size (difference in proportions)
    effect = p1 - p2

    if alternative == "two-sided":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt_2

        # Test statistic under alternative: (p̂1 - p̂2) / se_alt
        # Power = P(|Z| > z_alpha - |effect|/se_alt) where Z ~ N(0,1)
        standardized_effect = torch.abs(effect) / se_alt

        power = (1 - torch.erf((z_alpha - standardized_effect) / sqrt_2)) / 2 + (
            1 - torch.erf((z_alpha + standardized_effect) / sqrt_2)
        ) / 2

    elif alternative == "greater":
        # H1: p1 > p2
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2

        # Test statistic under alternative
        standardized_effect = effect / se_alt

        # Power = P(Z > z_alpha - effect/se_alt) where Z ~ N(0,1)
        power = (1 - torch.erf((z_alpha - standardized_effect) / sqrt_2)) / 2

    elif alternative == "less":
        # H1: p1 < p2
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2

        # Test statistic under alternative
        standardized_effect = effect / se_alt

        # Power = P(Z < -z_alpha - effect/se_alt) where Z ~ N(0,1)
        power = (1 + torch.erf((-z_alpha - standardized_effect) / sqrt_2)) / 2

    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    # Clamp power to [0, 1] range
    output = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
