import math

import torch
from torch import Tensor


def correlation_power(
    r: Tensor,
    sample_size: Tensor,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute statistical power for correlation tests.

    Statistical power is the probability of correctly rejecting a false null
    hypothesis that the population correlation is zero.

    This function is differentiable with respect to both r and sample_size
    parameters. While traditional correlation power analysis doesn't require
    gradients, differentiability enables integration into machine learning
    workflows where correlation strengths may be learned parameters.

    Parameters
    ----------
    r : Tensor
        Expected correlation coefficient. Can be a scalar or tensor.

    sample_size : Tensor
        Sample size (total number of observations).

    alpha : float, default=0.05
        Significance level (Type I error rate).

    alternative : str, default="two-sided"
        Type of alternative hypothesis. Options are "two-sided", "greater", or "less".

    out : Tensor, optional
        Output tensor. Default, `None`.

    Returns
    -------
    output : Tensor
        Statistical power (probability between 0 and 1).

    Examples
    --------
    >>> r = torch.tensor(0.3)
    >>> n = torch.tensor(50)
    >>> beignet.correlation_power(r, n)
    tensor(0.5704)
    """
    # Convert inputs to tensors if needed
    r = torch.as_tensor(r)
    sample_size = torch.as_tensor(sample_size)

    # Ensure both have the same dtype
    if r.dtype != sample_size.dtype:
        if r.dtype == torch.float64 or sample_size.dtype == torch.float64:
            r = r.to(torch.float64)
            sample_size = sample_size.to(torch.float64)
        else:
            r = r.to(torch.float32)
            sample_size = sample_size.to(torch.float32)

    # Fisher z-transformation of the correlation
    # z_r = 0.5 * ln((1 + r) / (1 - r))
    epsilon = 1e-7  # Small value to avoid division by zero
    r_clamped = torch.clamp(r, -1 + epsilon, 1 - epsilon)
    z_r = 0.5 * torch.log((1 + r_clamped) / (1 - r_clamped))

    # Standard error of Fisher z-transform
    # SE = 1 / sqrt(n - 3)
    se_z = 1.0 / torch.sqrt(sample_size - 3)

    # Test statistic under alternative hypothesis
    # z = z_r / SE
    z_stat = z_r / se_z

    # Critical values using standard normal approximation
    sqrt_2 = math.sqrt(2.0)

    if alternative == "two-sided":
        # Two-sided critical value
        z_alpha_2 = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=r.dtype)) * sqrt_2

        # Power = P(|Z| > z_alpha/2 | H1) where Z ~ N(z_stat, 1)
        # This is 1 - P(-z_alpha/2 < Z < z_alpha/2 | H1)
        # = 1 - [Φ(z_alpha/2 - z_stat) - Φ(-z_alpha/2 - z_stat)]
        cdf_upper = 0.5 * (1 + torch.erf((z_alpha_2 - z_stat) / sqrt_2))
        cdf_lower = 0.5 * (1 + torch.erf((-z_alpha_2 - z_stat) / sqrt_2))
        power = 1 - (cdf_upper - cdf_lower)

    elif alternative == "greater":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=r.dtype)) * sqrt_2

        # Power = P(Z > z_alpha | H1) = 1 - Φ(z_alpha - z_stat)
        power = 1 - 0.5 * (1 + torch.erf((z_alpha - z_stat) / sqrt_2))

    elif alternative == "less":
        z_alpha = torch.erfinv(torch.tensor(alpha, dtype=r.dtype)) * sqrt_2

        # Power = P(Z < z_alpha | H1) = Φ(z_alpha - z_stat)
        power = 0.5 * (1 + torch.erf((z_alpha - z_stat) / sqrt_2))

    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    # Clamp power to [0, 1] range
    output = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
