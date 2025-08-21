import math

import torch
from torch import Tensor


def correlation_sample_size(
    r: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute required sample size for correlation tests to achieve specified power.

    Given an expected correlation coefficient, desired power, and significance level,
    this function calculates the minimum sample size needed.

    This function is differentiable with respect to the r parameter.
    While traditional sample size calculations don't require gradients,
    differentiability can be useful when correlation strengths are learned
    parameters or when optimizing experimental designs in machine learning contexts.

    Parameters
    ----------
    r : Tensor
        Expected correlation coefficient. Can be a scalar or tensor.
    power : float, default=0.8
        Desired statistical power (probability of correctly rejecting false null).
    alpha : float, default=0.05
        Significance level (Type I error rate).
    alternative : str, default="two-sided"
        Type of alternative hypothesis. Options are "two-sided", "greater", or "less".

    Returns
    -------
    output : Tensor
        Required sample size (rounded up to nearest integer).

    Examples
    --------
    >>> r = torch.tensor(0.3)
    >>> beignet.correlation_sample_size(r, power=0.8)
    tensor(85)
    """
    # Convert inputs to tensors if needed
    r = torch.as_tensor(r)

    # Fisher z-transformation of the correlation
    # z_r = 0.5 * ln((1 + r) / (1 - r))
    epsilon = 1e-7  # Small value to avoid division by zero
    r_clamped = torch.clamp(r, -1 + epsilon, 1 - epsilon)
    z_r = 0.5 * torch.log((1 + r_clamped) / (1 - r_clamped))

    # Standard normal quantiles using erfinv
    sqrt_2 = math.sqrt(2.0)

    if alternative == "two-sided":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=r.dtype)) * sqrt_2
        z_beta = torch.erfinv(torch.tensor(power, dtype=r.dtype)) * sqrt_2
    elif alternative in ["greater", "less"]:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=r.dtype)) * sqrt_2
        z_beta = torch.erfinv(torch.tensor(power, dtype=r.dtype)) * sqrt_2
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    # Sample size formula for correlation test
    # For Fisher z-transform: SE = 1/sqrt(n-3), so n = ((z_alpha + z_beta) / z_r)^2 + 3
    # This comes from the power calculation: z_r / (1/sqrt(n-3)) = z_alpha + z_beta

    # Avoid division by very small correlations
    z_r_safe = torch.where(torch.abs(z_r) < 1e-6, torch.sign(z_r) * 1e-6, z_r)

    sample_size = ((z_alpha + z_beta) / torch.abs(z_r_safe)) ** 2 + 3

    # Round up to nearest integer
    output = torch.ceil(sample_size)

    if out is not None:
        out.copy_(output)
        return out

    return output
