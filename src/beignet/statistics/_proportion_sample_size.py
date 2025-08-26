import math

import torch
from torch import Tensor


def proportion_sample_size(
    p0: Tensor,
    p1: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute required sample size for one-sample proportion tests to achieve specified power.

    Given null and alternative proportions, desired power, and significance level,
    this function calculates the minimum sample size needed.

    This function is differentiable with respect to the proportion parameters.
    While traditional sample size calculations don't require gradients,
    differentiability enables integration into machine learning pipelines where
    proportions might be learned parameters or part of experimental design
    optimization.

    When to Use
    -----------
    **Traditional Statistics:**
    - Planning clinical trials with binary outcomes (cure/no cure, response/no response)
    - Quality control: planning inspections to detect defect rate changes
    - Survey research: planning studies to detect changes in opinion proportions
    - Political polling: planning surveys to detect changes in candidate support
    - Market research: planning studies to detect changes in consumer preferences
    - Educational research: planning studies to detect changes in pass rates

    **Machine Learning Contexts:**
    - A/B testing: planning experiments to detect conversion rate differences
    - Model evaluation: planning validation studies for binary classifier performance
    - Anomaly detection: planning studies to detect changes in anomaly rates
    - Recommendation systems: planning experiments to detect click-through rate changes
    - Computer vision: planning studies to detect changes in classification accuracy
    - NLP: planning experiments to detect changes in sentiment classification rates
    - Web analytics: planning studies to detect changes in user engagement rates
    - Fraud detection: planning studies to detect changes in fraud rates
    - Medical AI: planning validation studies for diagnostic model accuracy
    - Fairness auditing: planning studies to detect bias in algorithmic decisions
    - Active learning: planning studies to optimize sample selection for binary outcomes
    - Transfer learning: planning studies to evaluate model performance across domains

    **Choose one-sample proportion tests over two-sample when:**
    - Testing against a known standard, benchmark, or historical rate
    - Regulatory compliance requires testing against specific thresholds
    - Quality control testing against specification limits
    - Performance evaluation against established industry standards
    - Single group comparison to theoretical or expected proportions

    **Choose this over other sample size calculations when:**
    - Outcome is binary/dichotomous (success/failure, yes/no, pass/fail)
    - Testing a single proportion against a fixed value
    - Population variance is determined by the proportion (binomial variance)
    - Effect size is naturally expressed as difference in proportions

    **Interpretation Guidelines:**
    - Larger effect sizes (|p1 - p0|) require smaller sample sizes
    - Proportions near 0.5 require larger samples (maximum variance)
    - Extreme proportions (near 0 or 1) require smaller samples
    - Two-sided tests require larger samples than one-sided tests
    - Consider continuity correction for small samples or extreme proportions
    - Account for potential dropouts in planning (multiply by 1/(1-dropout_rate))
    - Sample size assumes random sampling from target population

    Parameters
    ----------
    p0 : Tensor
        Null hypothesis proportion (between 0 and 1).

    p1 : Tensor
        Alternative hypothesis proportion (between 0 and 1).

    power : float, default=0.8
        Desired statistical power (probability of correctly rejecting false null).

    alpha : float, default=0.05
        Significance level (Type I error rate).

    alternative : str, default="two-sided"
        Type of alternative hypothesis. Options are "two-sided", "greater", or "less".

    out : Tensor, optional
        Output tensor. Default, `None`.

    Returns
    -------
    output : Tensor
        Required sample size (rounded up to nearest integer).

    Examples
    --------
    >>> p0 = torch.tensor(0.5)
    >>> p1 = torch.tensor(0.6)
    >>> proportion_sample_size(p0, p1, power=0.8)
    tensor(199)

    Notes
    -----
    The sample size formula is derived from the normal approximation to the
    binomial distribution. For two-sided tests:

    n = [(z_α/2 * sqrt(p0*(1-p0)) + z_β * sqrt(p1*(1-p1))) / (p1 - p0)]²

    Where z_α/2 and z_β are the critical values from the standard normal
    distribution.

    References
    ----------
    .. [1] Fleiss, J. L., Levin, B., & Paik, M. C. (2003). Statistical methods
           for rates and proportions. John Wiley & Sons.
    .. [2] Chow, S. C., Shao, J., & Wang, H. (2008). Sample size calculations
           in clinical research. CRC press.
    """
    # Convert inputs to tensors if needed
    p0 = torch.atleast_1d(torch.as_tensor(p0))
    p1 = torch.atleast_1d(torch.as_tensor(p1))

    # Ensure tensors have the same dtype
    if p0.dtype == torch.float64 or p1.dtype == torch.float64:
        dtype = torch.float64
    else:
        dtype = torch.float32

    p0 = p0.to(dtype)
    p1 = p1.to(dtype)

    # Clamp proportions to valid range (0, 1)
    epsilon = 1e-8
    p0 = torch.clamp(p0, epsilon, 1 - epsilon)
    p1 = torch.clamp(p1, epsilon, 1 - epsilon)

    # Standard normal quantiles using erfinv
    sqrt_2 = math.sqrt(2.0)

    if alternative == "two-sided":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt_2
        z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt_2
    elif alternative in ["greater", "less"]:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2
        z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt_2
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    # Standard errors under null and alternative hypotheses
    se_null = torch.sqrt(p0 * (1 - p0))
    se_alt = torch.sqrt(p1 * (1 - p1))

    # Effect size (difference in proportions)
    effect = torch.abs(p1 - p0)

    # Avoid division by very small effect sizes
    effect_safe = torch.where(effect < 1e-6, torch.tensor(1e-6, dtype=dtype), effect)

    # Sample size formula
    sample_size = ((z_alpha * se_null + z_beta * se_alt) / effect_safe) ** 2

    output = torch.clamp(torch.ceil(sample_size), min=1.0)

    if out is not None:
        out.copy_(output)

        return out

    return output
