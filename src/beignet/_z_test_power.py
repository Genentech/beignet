"""Z-test power (one-sample z-test with known variance)."""

import math

import torch
from torch import Tensor


def z_test_power(
    effect_size: Tensor,
    sample_size: Tensor,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute statistical power for one-sample z-tests with known variance.

    This function calculates the probability of correctly rejecting the null
    hypothesis when the alternative hypothesis is true for a one-sample z-test
    where the population variance is known.

    This function is differentiable with respect to all tensor parameters.
    While traditional power analysis doesn't require gradients, differentiability
    enables integration into machine learning pipelines where effect sizes or
    sample sizes might be learned parameters or part of experimental design
    optimization.

    Parameters
    ----------
    effect_size : Tensor
        Standardized effect size (Cohen's d). This is the difference between
        the true population mean and the hypothesized mean, divided by the
        population standard deviation: d = (μ₁ - μ₀) / σ.
        Should be positive for "larger" alternative, negative for "smaller".
    sample_size : Tensor
        Sample size (number of observations).
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
    >>> sample_size = torch.tensor(30)
    >>> beignet.z_test_power(effect_size, sample_size)
    tensor(0.6985)

    Notes
    -----
    The power calculation is based on the standard normal distribution. For a
    one-sample z-test with effect size d and sample size n, the test statistic
    under the alternative hypothesis follows:

    Z = (X̄ - μ₀) / (σ / √n) ~ N(d√n, 1)

    Where μ₀ is the hypothesized mean, σ is the known population standard
    deviation, and d is the standardized effect size.

    For a two-sided test with significance level α:
    Power = P(|Z| > z_{α/2} | H₁) = P(Z > z_{α/2} - d√n) + P(Z < -z_{α/2} - d√n)

    For one-sided tests:
    - "larger": Power = P(Z > z_α - d√n)
    - "smaller": Power = P(Z < -z_α + d√n)

    Cohen's effect size interpretation:
    - Small effect: d = 0.20
    - Medium effect: d = 0.50
    - Large effect: d = 0.80

    References
    ----------
    .. [1] Cohen, J. (1988). Statistical power analysis for the behavioral
           sciences (2nd ed.). Erlbaum.
    .. [2] Cohen, J. (1992). A power primer. Psychological Bulletin, 112(1), 155-159.
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

    # Ensure positive sample size
    sample_size = torch.clamp(sample_size, min=1.0)

    # Calculate noncentrality parameter
    ncp = effect_size * torch.sqrt(sample_size)

    # Standard normal critical values using erfinv
    sqrt_2 = math.sqrt(2.0)

    if alternative == "two-sided":
        z_alpha_half = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt_2
        # Power = P(Z > z_{α/2} - δ) + P(Z < -z_{α/2} - δ)
        # where δ = d√n is the noncentrality parameter
        power_upper = (1 - torch.erf((z_alpha_half - ncp) / sqrt_2)) / 2
        power_lower = torch.erf((-z_alpha_half - ncp) / sqrt_2) / 2
        power = power_upper + power_lower
    elif alternative == "larger":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2
        # Power = P(Z > z_α - δ)
        power = (1 - torch.erf((z_alpha - ncp) / sqrt_2)) / 2
    elif alternative == "smaller":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2
        # Power = P(Z < -z_α + δ)
        power = torch.erf((-z_alpha + ncp) / sqrt_2) / 2
    else:
        raise ValueError(
            f"alternative must be 'two-sided', 'larger', or 'smaller', got {alternative}"
        )

    # Clamp power to [0, 1] range
    output = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
