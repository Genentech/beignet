import math

import torch
from torch import Tensor

from ._chi_squared_goodness_of_fit_power import chi_square_goodness_of_fit_power


def chi_square_goodness_of_fit_minimum_detectable_effect(
    sample_size: Tensor,
    df: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Minimum detectable Cohen's w for chi-square goodness-of-fit tests.

    Parameters
    ----------
    sample_size : Tensor
        Total sample size.
    df : Tensor
        Degrees of freedom (categories - 1).
    power : float, default=0.8
        Target power.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    Tensor
        Minimal Cohen's w achieving the requested power.
    """
    n0 = torch.as_tensor(sample_size)
    df0 = torch.as_tensor(df)
    scalar_out = n0.ndim == 0 and df0.ndim == 0
    n = torch.atleast_1d(n0)
    df = torch.atleast_1d(df0)
    dtype = (
        torch.float64
        if (n.dtype == torch.float64 or df.dtype == torch.float64)
        else torch.float32
    )
    n = torch.clamp(n.to(dtype), min=1.0)
    df = torch.clamp(df.to(dtype), min=1.0)

    # Initial guess using large-sample approximation: n ≈ ((zα+zβ)/w)^2
    sqrt2 = math.sqrt(2.0)
    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2
    w0 = torch.clamp((z_alpha + z_beta) / torch.sqrt(n), min=1e-8)

    # Bounded search via bisection using the implemented power function
    w_lo = torch.zeros_like(w0) + 1e-8
    w_hi = torch.clamp(2.0 * w0 + 1e-6, min=1e-6)

    # Ensure upper bound achieves at least the target power; expand as needed
    for _ in range(8):
        p_hi = chi_square_goodness_of_fit_power(w_hi, n, df, alpha)
        need_expand = p_hi < power
        if not torch.any(need_expand):
            break
        w_hi = torch.where(need_expand, w_hi * 2.0, w_hi)
        w_hi = torch.clamp(w_hi, max=torch.tensor(10.0, dtype=dtype))

    # Bisection
    w = (w_lo + w_hi) * 0.5
    for _ in range(24):
        p_mid = chi_square_goodness_of_fit_power(w, n, df, alpha)
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
