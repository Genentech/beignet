import math

import torch
from torch import Tensor


def two_one_sided_tests_two_sample_t_power(
    true_effect: Tensor,
    nobs1: Tensor,
    ratio: Tensor | float | None = None,
    low: Tensor = 0.0,
    high: Tensor = 0.0,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Compute power for two-sample Two One-Sided Tests (equivalence) under equal-variance t-test.

    Parameters
    ----------
    true_effect : Tensor
        Standardized true effect d = (μ₁ − μ₂)/σ_pooled.
    nobs1 : Tensor
        Group 1 sample size.
    ratio : Tensor or float, optional
        Ratio n2/n1. If None, uses 1.0.
    low : Tensor, default=0.0
        Lower equivalence margin (standardized).
    high : Tensor, default=0.0
        Upper equivalence margin (standardized).
    alpha : float, default=0.05
        Significance level per one-sided test.

    Returns
    -------
    Tensor
        Equivalence power.
    """
    d = torch.atleast_1d(torch.as_tensor(true_effect))
    n1 = torch.atleast_1d(torch.as_tensor(nobs1))
    if ratio is None:
        ratio_t = torch.tensor(1.0)
    else:
        ratio_t = torch.atleast_1d(torch.as_tensor(ratio))
    low = torch.atleast_1d(torch.as_tensor(low))
    high = torch.atleast_1d(torch.as_tensor(high))

    dtype = (
        torch.float64
        if any(t.dtype == torch.float64 for t in (d, n1, ratio_t, low, high))
        else torch.float32
    )
    d = d.to(dtype)
    n1 = n1.to(dtype)
    ratio_t = ratio_t.to(dtype)
    low = low.to(dtype)
    high = high.to(dtype)

    n1 = torch.clamp(n1, min=2.0)
    ratio_t = torch.clamp(ratio_t, min=0.1, max=10.0)
    n2 = n1 * ratio_t
    df = torch.clamp(n1 + n2 - 2, min=1.0)

    se_factor = torch.sqrt(1.0 / n1 + 1.0 / n2)
    ncp_low = (d - low) / torch.clamp(se_factor, min=1e-12)
    ncp_high = (d - high) / torch.clamp(se_factor, min=1e-12)

    sqrt2 = math.sqrt(2.0)
    z = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
    tcrit = z * torch.sqrt(1 + 1 / (2 * df))

    def power_greater(ncp: Tensor) -> Tensor:
        var = torch.where(
            df > 2,
            (df + ncp**2) / (df - 2),
            1 + ncp**2 / (2 * torch.clamp(df, min=1.0)),
        )
        std = torch.sqrt(var)
        zscore = (tcrit - ncp) / torch.clamp(std, min=1e-10)
        return 0.5 * (
            1 - torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
        )

    def power_less(ncp: Tensor) -> Tensor:
        var = torch.where(
            df > 2,
            (df + ncp**2) / (df - 2),
            1 + ncp**2 / (2 * torch.clamp(df, min=1.0)),
        )
        std = torch.sqrt(var)
        zscore = (-tcrit - ncp) / torch.clamp(std, min=1e-10)
        return 0.5 * (
            1 + torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
        )

    p_lower = power_greater(ncp_low)
    p_upper = power_less(ncp_high)
    power = torch.minimum(p_lower, p_upper)
    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)
        return out
    return power
