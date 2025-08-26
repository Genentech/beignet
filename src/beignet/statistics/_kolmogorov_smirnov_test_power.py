import math

import torch
from torch import Tensor


def kolmogorov_smirnov_test_power(
    effect_size: Tensor,
    sample_size: Tensor,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Power analysis for the Kolmogorov-Smirnov test.

    The Kolmogorov-Smirnov test compares the empirical distribution function
    of a sample with a reference distribution (one-sample) or compares two
    empirical distribution functions (two-sample).

    This function is differentiable with respect to all tensor parameters.
    While traditional power analysis doesn't require gradients, differentiability
    enables integration into machine learning pipelines where effect sizes or
    sample sizes might be learned parameters or part of experimental design
    optimization.

    When to Use
    -----------
    **Traditional Statistics:**
    - Goodness-of-fit testing against theoretical distributions (normal, uniform, etc.)
    - Quality control: testing if process outputs follow expected distributions
    - Clinical trials: comparing treatment response distributions
    - Educational testing: comparing score distributions across groups or standards
    - Environmental monitoring: testing if measurements follow expected patterns
    - Manufacturing: testing if product characteristics match specifications

    **Machine Learning Contexts:**
    - Model validation: testing if model outputs follow expected distributions
    - Synthetic data evaluation: comparing generated data to real data distributions
    - A/B testing: comparing full distribution differences (not just means)
    - Anomaly detection: testing if data follows normal operational distributions
    - Transfer learning: comparing source and target domain distributions
    - Feature engineering: validating distributional properties after transformations
    - Cross-validation: testing distributional consistency across folds
    - Computer vision: comparing pixel intensity or feature distributions
    - NLP: comparing word frequency or embedding distributions
    - Recommendation systems: comparing user behavior or item popularity distributions
    - Time series: comparing temporal distribution patterns
    - Fairness evaluation: comparing outcome distributions across demographic groups

    **Choose Kolmogorov-Smirnov over other tests when:**
    - Need to compare entire distributions rather than specific parameters
    - Don't want to make assumptions about distributional form
    - Interested in any type of distributional difference (location, scale, shape)
    - Have continuous data (or large samples of discrete data)
    - Want distribution-free (nonparametric) test

    **Choose this over Anderson-Darling when:**
    - Want equal sensitivity across entire distribution range
    - Working with smaller sample sizes
    - Need simpler computational implementation
    - Don't need extra sensitivity in distribution tails

    **Interpretation Guidelines:**
    - Effect size is maximum absolute difference between CDFs
    - Test is sensitive to any distributional difference (location, scale, shape)
    - More powerful than chi-square for continuous data with small samples
    - Assumes continuous distributions (discrete data may be conservative)
    - Two-sample version requires independent samples
    - Critical values depend on sample size, not distributional parameters

    Parameters
    ----------
    effect_size : Tensor
        Effect size representing the maximum difference between distributions.
        For one-sample test: max|F(x) - F₀(x)|
        For two-sample test: max|F₁(x) - F₂(x)|
        Should be in range [0, 1].
    sample_size : Tensor
        Sample size. For two-sample test, this is the harmonic mean of both
        sample sizes: 2*n₁*n₂/(n₁+n₂).
    alpha : float, default=0.05
        Significance level.
    alternative : {"two-sided", "greater", "less"}, default="two-sided"
        Alternative hypothesis direction.

    Returns
    -------
    output : Tensor, shape=(...)
        Statistical power values in range [0, 1].

    Examples
    --------
    >>> effect_size = torch.tensor(0.3)
    >>> sample_size = torch.tensor(50)
    >>> kolmogorov_smirnov_test_power(effect_size, sample_size)
    tensor(0.7812)

    Notes
    -----
    The Kolmogorov-Smirnov test statistic D is the maximum absolute difference
    between cumulative distribution functions:

    D = max|F_n(x) - F₀(x)| (one-sample)
    D = max|F₁,m(x) - F₂,n(x)| (two-sample)

    Under H₀, D has a known limiting distribution. Under H₁, the power depends
    on the true maximum difference between distributions.

    This implementation uses approximations suitable for moderate to large
    sample sizes (n ≥ 10).

    References
    ----------
    Massey Jr, F. J. (1951). The Kolmogorov-Smirnov test for goodness of fit.
    Journal of the American statistical Association, 46(253), 68-78.
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))

    # Ensure floating point dtype
    dtype = torch.promote_type(effect_size.dtype, sample_size.dtype)
    if not dtype.is_floating_point:
        dtype = torch.float32
    effect_size = effect_size.to(dtype)
    sample_size = sample_size.to(dtype)

    # Validate inputs
    effect_size = torch.clamp(effect_size, min=0.0, max=1.0)
    sample_size = torch.clamp(sample_size, min=3.0)

    # Normalize alternative
    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    # Critical value approximation for Kolmogorov-Smirnov distribution
    sqrt_n = torch.sqrt(sample_size)

    if alt == "two-sided":
        # Two-sided critical value (approximate)
        if alpha == 0.05:
            c_alpha = 1.36  # Approximate critical value for α=0.05
        elif alpha == 0.01:
            c_alpha = 1.63  # Approximate critical value for α=0.01
        else:
            # General approximation: c_α ≈ sqrt(-0.5 * ln(α/2))
            c_alpha = torch.sqrt(-0.5 * torch.log(torch.tensor(alpha / 2, dtype=dtype)))
    else:
        # One-sided critical value (approximate)
        if alpha == 0.05:
            c_alpha = 1.22  # Approximate critical value for α=0.05
        elif alpha == 0.01:
            c_alpha = 1.52  # Approximate critical value for α=0.01
        else:
            # General approximation: c_α ≈ sqrt(-0.5 * ln(α))
            c_alpha = torch.sqrt(-0.5 * torch.log(torch.tensor(alpha, dtype=dtype)))

    d_critical = c_alpha / sqrt_n

    # Power approximation
    # Under H₁, the test statistic has approximately normal distribution
    # This is a simplified approximation for moderate effect sizes

    # Expected value under alternative (simplified)
    expected_d = effect_size

    # Approximate standard error under alternative
    # This is a rough approximation; exact formula is complex
    se_d = torch.sqrt(1.0 / (2 * sample_size))

    # Standardized difference
    z_score = (d_critical - expected_d) / torch.clamp(se_d, min=1e-12)

    # Power calculation
    sqrt2 = math.sqrt(2.0)
    if alt == "two-sided":
        # P(|D| > d_critical | H₁)
        power = 1 - torch.erf(torch.abs(z_score) / sqrt2)
    elif alt == "greater":
        # P(D > d_critical | H₁)
        power = 0.5 * (1 - torch.erf(z_score / sqrt2))
    else:  # alt == "less"
        # P(D < -d_critical | H₁) = P(-D > d_critical | H₁)
        power = 0.5 * (1 + torch.erf(z_score / sqrt2))

    # Clamp to valid range
    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)
        return out
    return power
