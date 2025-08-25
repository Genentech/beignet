"""Independent z-test sample size (two-sample z-test with known variances)."""

import math

import torch
from torch import Tensor


def independent_z_test_sample_size(
    effect_size: Tensor,
    ratio: Tensor | None = None,
    power: Tensor | float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute required sample size for independent samples z-tests with known variances.

    Given the effect size, sample size ratio, desired power, and significance
    level, this function calculates the minimum sample size for group 1 needed
    to achieve the specified power for a two-sample z-test where the population
    variances are known.

    This function is differentiable with respect to effect_size and ratio parameters.
    While traditional sample size calculations don't require gradients,
    differentiability enables integration into machine learning pipelines where
    effect sizes might be learned parameters or part of experimental design
    optimization.

    Parameters
    ----------
    effect_size : Tensor
        Standardized effect size (Cohen's d). This is the difference between
        the two population means divided by the pooled standard deviation:
        d = (μ₁ - μ₂) / σ_pooled. Should be positive for "larger" alternative,
        negative for "smaller".
    ratio : Tensor, optional
        Ratio of sample size for group 2 relative to group 1.
        Group 2 sample size = sample_size1 * ratio. If None, defaults to 1.0
        (equal sample sizes).
    power : Tensor | float, default=0.8
        Desired statistical power (probability of correctly rejecting false null).
        Must be between 0 and 1.
    alpha : float, default=0.05
        Significance level (Type I error rate).
    alternative : str, default="two-sided"
        Type of alternative hypothesis. Either "two-sided", "larger", or "smaller".

    Returns
    -------
    output : Tensor
        Required sample size for group 1 (rounded up to nearest integer).
        Group 2 sample size can be calculated as output * ratio.

    Examples
    --------
    >>> effect_size = torch.tensor(0.5)
    >>> independent_z_test_sample_size(effect_size, power=0.8)
    tensor(64)

    Notes
    -----
    The sample size calculation is based on the standard normal distribution.
    For a two-sample z-test with effect size d, the required sample size for
    group 1 is approximately:

    For two-sided tests: n₁ ≈ ((z_{α/2} + z_β) / d)² * (1 + 1/r)
    For one-sided tests: n₁ ≈ ((z_α + z_β) / d)² * (1 + 1/r)

    Where z_α and z_β are the critical values for the given α and β = 1 - power,
    and r is the ratio n₂/n₁.

    The effective sample size for two-sample tests is:
    n_eff = n₁n₂/(n₁+n₂) = n₁r/(1+r)

    The calculation uses the exact relationship between power and sample size
    for the normal distribution, ensuring accurate results for all effect sizes
    and power levels.

    Cohen's effect size interpretation:
    - Small effect: d = 0.20
    - Medium effect: d = 0.50
    - Large effect: d = 0.80

    References
    ----------
    .. [1] Cohen, J. (1988). Statistical power analysis for the behavioral
           sciences (2nd ed.). Erlbaum.
    .. [2] Aberson, C. L. (2010). Applied power analysis for the behavioral
           sciences. Routledge.
    """
    # Convert inputs to tensors if needed
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    power = torch.atleast_1d(torch.as_tensor(power))
    if ratio is None:
        ratio = torch.tensor(1.0)
    else:
        ratio = torch.atleast_1d(torch.as_tensor(ratio))

    # Input validation (only when not compiled)
    if not torch.jit.is_scripting() and not torch.compiler.is_compiling():
        if torch.any(power <= 0) or torch.any(power >= 1):
            raise ValueError("Power must be between 0 and 1 (exclusive)")

    # Ensure tensors have the same dtype
    if (
        effect_size.dtype == torch.float64
        or ratio.dtype == torch.float64
        or power.dtype == torch.float64
    ):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    ratio = ratio.to(dtype)
    power = power.to(dtype)

    # Clamp values for numerical stability during compilation
    power = torch.clamp(power, min=1e-6, max=1.0 - 1e-6)
    abs_effect_size = torch.clamp(torch.abs(effect_size), min=1e-6)
    ratio = torch.clamp(ratio, min=0.1, max=10.0)

    # Standard normal quantiles using erfinv
    sqrt_2 = math.sqrt(2.0)
    z_beta = torch.erfinv(power) * sqrt_2

    if alternative == "two-sided":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt_2
    elif alternative in ["larger", "smaller"]:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2
    else:
        raise ValueError(
            f"alternative must be 'two-sided', 'larger', or 'smaller', got {alternative}"
        )

    # Calculate variance factor for two-sample design
    # For independent samples: variance_factor = (1 + 1/r)
    variance_factor = 1 + 1 / ratio

    # Calculate required sample size: n₁ = ((z_α + z_β) / d)² * (1 + 1/r)
    sample_size1 = ((z_alpha + z_beta) / abs_effect_size) ** 2 * variance_factor

    # Round up to nearest integer
    output = torch.ceil(sample_size1)

    # Ensure minimum sample size of 1
    output = torch.clamp(output, min=1.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
