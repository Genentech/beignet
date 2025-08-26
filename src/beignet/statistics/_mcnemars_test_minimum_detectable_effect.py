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
    r"""
    """
    d0 = torch.as_tensor(discordant_rate)

    n0 = torch.as_tensor(sample_size)

    scalar_out = d0.ndim == 0 and n0.ndim == 0

    d = torch.atleast_1d(d0)

    sample_size = torch.atleast_1d(n0)

    dtype = (
        torch.float64
        if any(t.dtype == torch.float64 for t in (d, sample_size))
        else torch.float32
    )
    d = torch.clamp(d.to(dtype), min=1e-8, max=1.0)

    sample_size = torch.clamp(sample_size.to(dtype), min=1.0)

    sqrt2 = math.sqrt(2.0)

    def z_of(prob: float) -> Tensor:
    r"""
    """
        pt = torch.tensor(prob, dtype=dtype)

        eps = torch.finfo(dtype).eps

        pt = torch.clamp(pt, min=eps, max=1 - eps)
        return sqrt2 * torch.erfinv(2.0 * pt - 1.0)

    z_alpha = z_of(1 - alpha / 2) if two_sided else z_of(1 - alpha)

    z_beta = z_of(power)

    diff0 = (z_alpha + z_beta) * torch.sqrt(
        torch.clamp(d / sample_size, min=torch.finfo(dtype).eps),
    )
    max_allowed = torch.minimum(d, torch.tensor(1.0, dtype=dtype))

    min_allowed = torch.tensor(1e-8, dtype=dtype)

    diff0 = torch.clamp(diff0, min_allowed, max_allowed)

    lo = torch.maximum(torch.zeros_like(diff0) + 1e-8, min_allowed)

    hi = torch.clamp(2.0 * diff0, min_allowed, max_allowed)

    for _ in range(8):
        p01_hi = torch.clamp((d + hi) / 2.0, 0.0, 1.0)

        p10_hi = torch.clamp(d - p01_hi, 0.0, 1.0)

        p_hi = mcnemars_test_power(
            p01_hi,
            p10_hi,
            sample_size,
            alpha=alpha,
            two_sided=two_sided,
        )
        need_expand = p_hi < power
        if not torch.any(need_expand):
            break
        hi = torch.where(need_expand, torch.clamp(hi * 2.0, max=max_allowed), hi)

    x = (lo + hi) * 0.5
    for _ in range(24):
        p01_mid = torch.clamp((d + x) / 2.0, 0.0, 1.0)

        p10_mid = torch.clamp(d - p01_mid, 0.0, 1.0)

        p_mid = mcnemars_test_power(
            p01_mid,
            p10_mid,
            sample_size,
            alpha=alpha,
            two_sided=two_sided,
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
