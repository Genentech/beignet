import math

import torch
from torch import Tensor


def two_one_sided_tests_one_sample_t_power(
    true_effect: Tensor,
    sample_size: Tensor,
    low: Tensor,
    high: Tensor,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Compute power for one-sample Two One-Sided Tests (equivalence) with standardized margins.

    Parameters
    ----------
    true_effect : Tensor
        Standardized true effect d = (μ - μ0)/σ.
    sample_size : Tensor
        Sample size n.
    low : Tensor
        Lower equivalence margin (standardized).
    high : Tensor
        Upper equivalence margin (standardized).
    alpha : float, default=0.05
        Significance level for each one-sided test.

    Returns
    -------
    Tensor
        Equivalence power (probability both one-sided tests reject).
    """
    d = torch.atleast_1d(torch.as_tensor(true_effect))
    n = torch.atleast_1d(torch.as_tensor(sample_size))
    low = torch.atleast_1d(torch.as_tensor(low))
    high = torch.atleast_1d(torch.as_tensor(high))

    dtype = (
        torch.float64
        if any(t.dtype == torch.float64 for t in (d, n, low, high))
        else torch.float32
    )
    d = d.to(dtype)
    n = n.to(dtype)
    low = low.to(dtype)
    high = high.to(dtype)

    n = torch.clamp(n, min=2.0)
    df = n - 1
    ncp_low = (d - low) * torch.sqrt(n)
    ncp_high = (d - high) * torch.sqrt(n)

    sqrt2 = math.sqrt(2.0)
    z = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
    tcrit = z * torch.sqrt(1 + 1 / (2 * df))

    # Approximate noncentral t by normal
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
