import math

import torch
from torch import Tensor

from ._mcnemars_test_power import mcnemars_test_power


def mcnemars_test_minimum_detectable_effect(
    discordant_rate: Tensor,
    sample_size: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    two_sided: bool = True,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Minimum detectable absolute difference |p01 - p10| for McNemar's test.

    Parameters
    ----------
    discordant_rate : Tensor
        Total probability of discordant pairs, p01 + p10 (in (0,1]).
    sample_size : Tensor
        Number of paired observations (n).
    power : float, default=0.8
        Target power.
    alpha : float, default=0.05
        Significance level.
    two_sided : bool, default=True
        Whether the test is two-sided.

    Returns
    -------
    Tensor
        Minimal |p01 - p10| achieving the requested power, given p01 + p10.

    Notes
    -----
    Under the normal approximation used in power/sample size, the noncentrality
    depends on |p01 - p10| only through (p01 + p10); thus
    |p01 - p10|_min ≈ (z_α + z_β) * sqrt((p01+p10)/n).
    """
    d0 = torch.as_tensor(discordant_rate)
    n0 = torch.as_tensor(sample_size)
    scalar_out = d0.ndim == 0 and n0.ndim == 0
    d = torch.atleast_1d(d0)
    n = torch.atleast_1d(n0)
    dtype = (
        torch.float64
        if any(t.dtype == torch.float64 for t in (d, n))
        else torch.float32
    )
    d = torch.clamp(d.to(dtype), min=1e-8, max=1.0)
    n = torch.clamp(n.to(dtype), min=1.0)

    sqrt2 = math.sqrt(2.0)

    def z_of(prob: float) -> Tensor:
        pt = torch.tensor(prob, dtype=dtype)
        eps = torch.finfo(dtype).eps
        pt = torch.clamp(pt, min=eps, max=1 - eps)
        return sqrt2 * torch.erfinv(2.0 * pt - 1.0)

    z_alpha = z_of(1 - alpha / 2) if two_sided else z_of(1 - alpha)
    z_beta = z_of(power)
    # From ncp = sqrt(n/d) * |p01 - p10|
    # Initial guess
    diff0 = (z_alpha + z_beta) * torch.sqrt(
        torch.clamp(d / n, min=torch.finfo(dtype).eps)
    )
    max_allowed = torch.minimum(d, torch.tensor(1.0, dtype=dtype))
    min_allowed = torch.tensor(1e-8, dtype=dtype)
    diff0 = torch.clamp(diff0, min_allowed, max_allowed)

    # Bounded search via bisection using power function; constrain diff in [0, min(d,1)]
    lo = torch.maximum(torch.zeros_like(diff0) + 1e-8, min_allowed)
    hi = torch.clamp(2.0 * diff0, min_allowed, max_allowed)

    # Ensure upper bound achieves the target power
    for _ in range(8):
        p01_hi = torch.clamp((d + hi) / 2.0, 0.0, 1.0)
        p10_hi = torch.clamp(d - p01_hi, 0.0, 1.0)
        p_hi = mcnemars_test_power(p01_hi, p10_hi, n, alpha=alpha, two_sided=two_sided)
        need_expand = p_hi < power
        if not torch.any(need_expand):
            break
        hi = torch.where(need_expand, torch.clamp(hi * 2.0, max=max_allowed), hi)

    # Bisection refinement
    x = (lo + hi) * 0.5
    for _ in range(24):
        p01_mid = torch.clamp((d + x) / 2.0, 0.0, 1.0)
        p10_mid = torch.clamp(d - p01_mid, 0.0, 1.0)
        p_mid = mcnemars_test_power(
            p01_mid, p10_mid, n, alpha=alpha, two_sided=two_sided
        )
        go_right = p_mid < power
        lo = torch.where(go_right, x, lo)
        hi = torch.where(go_right, hi, x)
        x = (lo + hi) * 0.5
    abs_diff = x

    if scalar_out:
        diff_s = abs_diff.reshape(())
        if out is not None:
            out.copy_(diff_s)
            return out
        return diff_s
    else:
        if out is not None:
            out.copy_(abs_diff)
            return out
        return abs_diff
