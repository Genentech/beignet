import math

import torch
from torch import Tensor


def proportion_two_sample_sample_size(
    p1: Tensor,
    p2: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    ratio: float = 1.0,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute required sample size for two-sample proportion tests to achieve specified power.

    Given proportions for two groups, desired power, and significance level,
    this function calculates the minimum sample size needed for each group.

    This function is differentiable with respect to the proportion parameters.
    While traditional sample size calculations don't require gradients,
    differentiability enables integration into machine learning pipelines where
    proportions might be learned parameters or part of experimental design
    optimization.

    Parameters
    ----------
    p1 : Tensor
        Proportion in group 1 (between 0 and 1).

    p2 : Tensor
        Proportion in group 2 (between 0 and 1).

    power : float, default=0.8
        Desired statistical power (probability of correctly detecting the difference).

    alpha : float, default=0.05
        Significance level (Type I error rate).

    alternative : str, default="two-sided"
        Type of alternative hypothesis. Options are "two-sided", "greater", or "less".

    ratio : float, default=1.0
        Ratio of sample sizes n2/n1. Default is 1.0 for equal sample sizes.

    out : Tensor, optional
        Output tensor. Default, `None`.

    Returns
    -------
    output : Tensor
        Required sample size for group 1 (rounded up to nearest integer).
        Sample size for group 2 can be calculated as n2 = ratio * n1.

    Examples
    --------
    >>> p1 = torch.tensor(0.5)
    >>> p2 = torch.tensor(0.6)
    >>> beignet.proportion_two_sample_sample_size(p1, p2, power=0.8)
    tensor(387)

    Notes
    -----
    The sample size formula for equal sample sizes (ratio=1) is:

    n = 2 * [(z_α/2 * sqrt(2*p̄*(1-p̄)) + z_β * sqrt(p1*(1-p1) + p2*(1-p2))) / (p1 - p2)]²

    Where p̄ = (p1 + p2) / 2 is the average proportion.

    For unequal sample sizes, the formula becomes more complex and involves
    the ratio parameter.

    References
    ----------
    .. [1] Fleiss, J. L., Levin, B., & Paik, M. C. (2003). Statistical methods
           for rates and proportions. John Wiley & Sons.
    .. [2] Chow, S. C., Shao, J., & Wang, H. (2008). Sample size calculations
           in clinical research. CRC press.
    """
    # Convert inputs to tensors if needed
    p1 = torch.as_tensor(p1)
    p2 = torch.as_tensor(p2)

    # Ensure tensors have the same dtype
    if p1.dtype == torch.float64 or p2.dtype == torch.float64:
        dtype = torch.float64
    else:
        dtype = torch.float32

    p1 = p1.to(dtype)
    p2 = p2.to(dtype)
    ratio = torch.tensor(ratio, dtype=dtype)

    # Clamp proportions to valid range (0, 1)
    epsilon = 1e-8
    p1 = torch.clamp(p1, epsilon, 1 - epsilon)
    p2 = torch.clamp(p2, epsilon, 1 - epsilon)

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

    # Pooled proportion estimate for null hypothesis
    p_pooled = (p1 + p2 * ratio) / (1 + ratio)
    p_pooled = torch.clamp(p_pooled, epsilon, 1 - epsilon)

    # Effect size (absolute difference in proportions)
    effect = torch.abs(p1 - p2)

    # Avoid division by very small effect sizes
    effect_safe = torch.where(effect < 1e-6, torch.tensor(1e-6, dtype=dtype), effect)

    # Variance components
    # Under null hypothesis (pooled variance)
    var_null = p_pooled * (1 - p_pooled) * (1 + 1 / ratio)

    # Under alternative hypothesis (separate variances)
    var_alt = p1 * (1 - p1) + p2 * (1 - p2) / ratio

    # Sample size formula
    numerator = z_alpha * torch.sqrt(var_null) + z_beta * torch.sqrt(var_alt)
    sample_size = (numerator / effect_safe) ** 2

    # Round up to nearest integer
    output = torch.ceil(sample_size)

    # Ensure minimum sample size of 1
    output = torch.clamp(output, min=1.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
