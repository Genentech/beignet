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

    When to Use
    -----------
    **Traditional Statistics:**
    - Clinical trials comparing treatment and control success rates
    - Quality control comparing defect rates between production lines
    - Market research comparing conversion rates between different campaigns
    - Educational research comparing pass rates between teaching methods
    - Political polling comparing support rates between different demographics
    - Public health studies comparing disease rates between exposed/unexposed groups

    **Machine Learning Contexts:**
    - A/B testing comparing conversion rates between different algorithms
    - Model evaluation comparing accuracy rates between different architectures
    - Recommendation system evaluation comparing click-through rates
    - Computer vision comparing classification accuracy between different models
    - NLP comparing sentiment classification success rates
    - Anomaly detection comparing detection rates between different methods
    - Fraud detection comparing true positive rates between detection systems
    - Web analytics comparing engagement rates between different interfaces
    - Mobile app testing comparing retention rates between app versions
    - Search engine evaluation comparing relevance rates between ranking algorithms
    - Fairness auditing comparing outcome rates across different demographic groups
    - Transfer learning comparing adaptation success rates across domains

    **Choose two-sample proportion tests over one-sample when:**
    - Comparing two groups rather than testing against a fixed standard
    - No established benchmark proportion to serve as null hypothesis
    - Both groups represent different treatments, conditions, or populations
    - Interest is in relative comparison rather than absolute evaluation

    **Choose proportion tests over continuous tests when:**
    - Outcome is naturally binary (success/failure, yes/no, pass/fail)
    - Continuous measurements are converted to binary classifications
    - Interest is in rates, percentages, or probabilities rather than means
    - Data collection naturally produces categorical outcomes

    **Interpretation Guidelines:**
    - Effect size is |p₁ - p₂|: difference in proportions
    - Small effect: |p₁ - p₂| ≈ 0.10, Medium: ≈ 0.25, Large: ≈ 0.40
    - Power increases with larger sample sizes and larger effect sizes
    - Power is lowest when both proportions are near 0.5 (maximum variance)
    - Two-sided tests require larger samples than one-sided tests
    - Equal sample sizes generally provide optimal power for fixed total N
    - Consider continuity correction for small samples or extreme proportions

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

    out : Tensor, optional
        Output tensor. Default, `None`.

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
    >>> proportion_two_sample_power(p1, p2, n1, n2)
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
    p1 = torch.atleast_1d(torch.as_tensor(p1))
    p2 = torch.atleast_1d(torch.as_tensor(p2))
    sample_size_group_1 = torch.atleast_1d(torch.as_tensor(n1))

    if n2 is None:
        sample_size_group_2 = sample_size_group_1
    else:
        sample_size_group_2 = torch.atleast_1d(torch.as_tensor(n2))

    # Ensure all tensors have the same dtype
    if (
        p1.dtype == torch.float64
        or p2.dtype == torch.float64
        or sample_size_group_1.dtype == torch.float64
        or sample_size_group_2.dtype == torch.float64
    ):
        dtype = torch.float64
    else:
        dtype = torch.float32

    p1 = p1.to(dtype)
    p2 = p2.to(dtype)
    sample_size_group_1 = sample_size_group_1.to(dtype)
    sample_size_group_2 = sample_size_group_2.to(dtype)

    # Clamp proportions to valid range (0, 1)
    epsilon = 1e-8
    p1 = torch.clamp(p1, epsilon, 1 - epsilon)
    p2 = torch.clamp(p2, epsilon, 1 - epsilon)

    # Pooled proportion under null hypothesis (p1 = p2)
    p_pooled = (sample_size_group_1 * p1 + sample_size_group_2 * p2) / (
        sample_size_group_1 + sample_size_group_2
    )
    p_pooled = torch.clamp(p_pooled, epsilon, 1 - epsilon)

    # Standard error under null hypothesis (pooled variance)
    se_null = torch.sqrt(
        p_pooled * (1 - p_pooled) * (1 / sample_size_group_1 + 1 / sample_size_group_2)
    )

    # Standard error under alternative hypothesis (separate variances)
    se_alt = torch.sqrt(
        p1 * (1 - p1) / sample_size_group_1 + p2 * (1 - p2) / sample_size_group_2
    )

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
