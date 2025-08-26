import math

import torch
from torch import Tensor

from ._chi_squared_goodness_of_fit_power import chi_square_goodness_of_fit_power


def chi_square_goodness_of_fit_minimum_detectable_effect(
    sample_size: Tensor,
    degrees_of_freedom: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    sample_size_0 = torch.as_tensor(sample_size)

    degrees_of_freedom_0 = torch.as_tensor(degrees_of_freedom)

    scalar_out = sample_size_0.ndim == 0 and degrees_of_freedom_0.ndim == 0

    sample_size = torch.atleast_1d(sample_size_0)

    degrees_of_freedom = torch.atleast_1d(degrees_of_freedom_0)

    dtype = (
        torch.float64
        if (
            sample_size.dtype == torch.float64
            or degrees_of_freedom.dtype == torch.float64
        )
        else torch.float32
    )
    sample_size = torch.clamp(sample_size.to(dtype), min=1.0)

    degrees_of_freedom = torch.clamp(degrees_of_freedom.to(dtype), min=1.0)

    sqrt2 = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2

    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2

    w0 = torch.clamp((z_alpha + z_beta) / torch.sqrt(sample_size), min=1e-8)

    w_lo = torch.zeros_like(w0) + 1e-8

    w_hi = torch.clamp(2.0 * w0 + 1e-6, min=1e-6)

    for _ in range(8):
        p_hi = chi_square_goodness_of_fit_power(
            w_hi, sample_size, degrees_of_freedom, alpha
        )
        need_expand = p_hi < power
        if not torch.any(need_expand):
            break
        w_hi = torch.where(need_expand, w_hi * 2.0, w_hi)

        w_hi = torch.clamp(w_hi, max=torch.tensor(10.0, dtype=dtype))

    w = (w_lo + w_hi) * 0.5
    for _ in range(24):
        p_mid = chi_square_goodness_of_fit_power(
            w, sample_size, degrees_of_freedom, alpha
        )
        go_right = p_mid < power

        w_lo = torch.where(go_right, w, w_lo)
        w_hi = torch.where(go_right, w_hi, w)

        w = (w_lo + w_hi) * 0.5

    out_t = torch.clamp(w, min=0.0)
    if scalar_out:
        out_scalar = out_t.reshape(())
        if out is not None:
            out.copy_(out_scalar)
            return out
        return out_scalar
    else:
        if out is not None:
            out.copy_(out_t)
            return out
        return out_t
