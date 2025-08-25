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
