"""Z-test power (one-sample z-test with known variance)."""

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
    >>> z_test_power(effect_size, sample_size)
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
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))

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

    # Normalize alternative
    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError(
            f"alternative must be 'two-sided', 'greater', or 'less', got {alternative}"
        )

    # Standard normal critical values
    sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=dtype))

    def z_of(p):
        pt = torch.as_tensor(p, dtype=dtype)
        eps = torch.finfo(dtype).eps
        pt = torch.clamp(pt, min=eps, max=1 - eps)
        return sqrt2 * torch.erfinv(2.0 * pt - 1.0)

    if alt == "two-sided":
        z_alpha_half = z_of(1 - alpha / 2)
        # Power = P(Z > z_{α/2} - δ) + P(Z < -z_{α/2} - δ)
        # where δ = d√n is the noncentrality parameter
        power_upper = 0.5 * (
            1 - torch.erf((z_alpha_half - ncp) / torch.sqrt(torch.tensor(2.0)))
        )
        power_lower = 0.5 * (
            1 + torch.erf((-z_alpha_half - ncp) / torch.sqrt(torch.tensor(2.0)))
        )
        power = power_upper + power_lower
    elif alt == "greater":
        z_alpha = z_of(1 - alpha)
        # Power = P(Z > z_α - δ)
        power = 0.5 * (1 - torch.erf((z_alpha - ncp) / torch.sqrt(torch.tensor(2.0))))
    else:  # alt == 'less'
        z_alpha = z_of(1 - alpha)
        # Power = P(Z < -z_α - δ) = Φ(-z_α - ncp)
        power = 0.5 * (1 + torch.erf((-z_alpha - ncp) / torch.sqrt(torch.tensor(2.0))))

    # Clamp power to [0, 1] range
    output = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(output)

        return out

    return output
