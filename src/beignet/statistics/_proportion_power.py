import math

import torch
from torch import Tensor


def proportion_power(
    p0: Tensor,
    p1: Tensor,
    sample_size: Tensor,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute statistical power for one-sample proportion tests.

    Given a null hypothesis proportion, alternative proportion, and sample size,
    this function calculates the probability of correctly rejecting the false
    null hypothesis (statistical power).

    This function is differentiable with respect to all tensor parameters.
    While traditional power analysis doesn't require gradients, differentiability
    enables integration into machine learning pipelines where proportions or
    sample sizes might be learned parameters or part of experimental design
    optimization.

    Parameters
    ----------
    p0 : Tensor
        Null hypothesis proportion (between 0 and 1).

    p1 : Tensor
        Alternative hypothesis proportion (between 0 and 1).

    sample_size : Tensor
        Sample size for the test.

    alpha : float, default=0.05
        Significance level (Type I error rate).

    alternative : str, default="two-sided"
        Type of alternative hypothesis. Options are "two-sided", "greater", or "less".

    out : Tensor, optional
        Output tensor. Default, `None`.

    Returns
    -------
    output : Tensor
        Statistical power (probability of correctly rejecting false null hypothesis).

    Examples
    --------
    >>> p0 = torch.tensor(0.5)
    >>> p1 = torch.tensor(0.6)
    >>> n = torch.tensor(100)
    >>> proportion_power(p0, p1, n)
    tensor(0.7139)

    Notes
    -----
    The test statistic follows a normal distribution under the null hypothesis:
    Z = (p̂ - p0) / sqrt(p0 * (1 - p0) / n)

    Under the alternative hypothesis, the test statistic has mean:
    μ = (p1 - p0) / sqrt(p0 * (1 - p0) / n)

    And standard deviation:
    σ = sqrt(p1 * (1 - p1) / (p0 * (1 - p0)))

    References
    ----------
    .. [1] Fleiss, J. L., Levin, B., & Paik, M. C. (2003). Statistical methods
           for rates and proportions. John Wiley & Sons.
    """
    # Convert inputs to tensors if needed
    p0 = torch.atleast_1d(torch.as_tensor(p0))
    p1 = torch.atleast_1d(torch.as_tensor(p1))
    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))

    # Ensure all tensors have the same dtype
    if (
        p0.dtype == torch.float64
        or p1.dtype == torch.float64
        or sample_size.dtype == torch.float64
    ):
        dtype = torch.float64
    else:
        dtype = torch.float32

    p0 = p0.to(dtype)
    p1 = p1.to(dtype)
    sample_size = sample_size.to(dtype)

    # Clamp proportions to valid range (0, 1)
    epsilon = 1e-8
    p0 = torch.clamp(p0, epsilon, 1 - epsilon)
    p1 = torch.clamp(p1, epsilon, 1 - epsilon)

    # Standard normal quantiles using erfinv
    sqrt_2 = math.sqrt(2.0)

    # Standard error under null hypothesis (pooled)
    se_null = torch.sqrt(p0 * (1 - p0) / sample_size)

    # Standard error under alternative hypothesis
    se_alt = torch.sqrt(p1 * (1 - p1) / sample_size)

    # Effect size (difference in proportions divided by null SE)
    effect = (p1 - p0) / se_null

    if alternative == "two-sided":
        z_alpha_half = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt_2

        # Adjust for different variance under alternative
        variance_ratio = se_alt / se_null
        adjusted_effect = effect / variance_ratio

        # Power = P(|Z| > z_alpha/2 - adjusted_effect) where Z ~ N(0,1) under alternative
        power = (1 - torch.erf((z_alpha_half - adjusted_effect) / sqrt_2)) / 2 + (
            1 - torch.erf((z_alpha_half + adjusted_effect) / sqrt_2)
        ) / 2

    elif alternative == "greater":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2

        # Adjust for different variance under alternative
        variance_ratio = se_alt / se_null
        adjusted_effect = effect / variance_ratio

        # Power = P(Z > z_alpha - adjusted_effect) where Z ~ N(0,1) under alternative
        power = (1 - torch.erf((z_alpha - adjusted_effect) / sqrt_2)) / 2

    elif alternative == "less":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2

        # Adjust for different variance under alternative
        variance_ratio = se_alt / se_null
        adjusted_effect = effect / variance_ratio

        # Power = P(Z < -z_alpha - adjusted_effect) where Z ~ N(0,1) under alternative
        power = (1 + torch.erf((-z_alpha - adjusted_effect) / sqrt_2)) / 2

    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    # Clamp power to [0, 1] range
    output = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
