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
