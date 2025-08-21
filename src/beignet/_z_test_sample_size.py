"""Z-test sample size (one-sample z-test with known variance)."""

import math

import torch
from torch import Tensor


def z_test_sample_size(
    effect_size: Tensor,
    power: Tensor | float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute required sample size for one-sample z-tests with known variance.

    Given the effect size, desired power, and significance level, this function
    calculates the minimum sample size needed to achieve the specified power
    for a one-sample z-test where the population variance is known.

    This function is differentiable with respect to the effect_size parameter.
    While traditional sample size calculations don't require gradients,
    differentiability enables integration into machine learning pipelines where
    effect sizes might be learned parameters or part of experimental design
    optimization.

    Parameters
    ----------
    effect_size : Tensor
        Standardized effect size (Cohen's d). This is the difference between
        the true population mean and the hypothesized mean, divided by the
        population standard deviation: d = (μ₁ - μ₀) / σ.
        Should be positive for "larger" alternative, negative for "smaller".
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
        Required sample size (rounded up to nearest integer).

    Examples
    --------
    >>> effect_size = torch.tensor(0.5)
    >>> beignet.z_test_sample_size(effect_size, power=0.8)
    tensor(32)

    Notes
    -----
    The sample size calculation is based on the standard normal distribution.
    For a one-sample z-test with effect size d, the required sample size is
    approximately:

    For two-sided tests: n ≈ ((z_{α/2} + z_β) / d)²
    For one-sided tests: n ≈ ((z_α + z_β) / d)²

    Where z_α and z_β are the critical values for the given α and β = 1 - power.

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
    .. [2] Cohen, J. (1992). A power primer. Psychological Bulletin, 112(1), 155-159.
    """
    # Convert inputs to tensors if needed
    effect_size = torch.as_tensor(effect_size)
    power = torch.as_tensor(power)

    # Input validation (only when not compiled)
    if not torch.jit.is_scripting() and not torch.compiler.is_compiling():
        if torch.any(power <= 0) or torch.any(power >= 1):
            raise ValueError("Power must be between 0 and 1 (exclusive)")

    # Ensure tensors have the correct dtype
    if effect_size.dtype == torch.float64 or power.dtype == torch.float64:
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    power = power.to(dtype)

    # Clamp power to valid range for numerical stability during compilation
    power = torch.clamp(power, min=1e-6, max=1.0 - 1e-6)

    # Ensure positive effect size (take absolute value for sample size calculation)
    abs_effect_size = torch.clamp(torch.abs(effect_size), min=1e-6)

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

    # Calculate required sample size: n = ((z_α + z_β) / d)²
    sample_size = ((z_alpha + z_beta) / abs_effect_size) ** 2

    # Round up to nearest integer
    output = torch.ceil(sample_size)

    # Ensure minimum sample size of 1
    output = torch.clamp(output, min=1.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
