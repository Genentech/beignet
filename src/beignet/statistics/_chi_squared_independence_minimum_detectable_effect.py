import math

import torch
from torch import Tensor

from ._chi_squared_independence_power import chi_square_independence_power


def chi_square_independence_minimum_detectable_effect(
    sample_size: Tensor,
    rows: Tensor,
    cols: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    sample_size : Tensor
        Sample size.
    rows : Tensor
        Rows parameter.
    cols : Tensor
        Cols parameter.
    power : float, default 0.8
        Statistical power.
    alpha : float, default 0.05
        Type I error rate.
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        Effect size.
    """

    sample_size_0 = torch.as_tensor(sample_size)

    r0 = torch.as_tensor(rows)
    c0 = torch.as_tensor(cols)

    scalar_out = sample_size_0.ndim == 0 and r0.ndim == 0 and c0.ndim == 0

    sample_size = torch.atleast_1d(sample_size_0)

    r = torch.atleast_1d(r0)
    c = torch.atleast_1d(c0)

    dtype = torch.float32
    for tensor in (sample_size, r, c):
        dtype = torch.promote_types(dtype, tensor.dtype)
    sample_size = torch.clamp(sample_size.to(dtype), min=1.0)

    r = torch.clamp(r.to(dtype), min=2.0)
    c = torch.clamp(c.to(dtype), min=2.0)

    square_root_two = math.sqrt(2.0)

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * square_root_two

    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * square_root_two

    w0 = torch.clamp((z_alpha + z_beta) / torch.sqrt(sample_size), min=1e-8)

    w_lo = torch.zeros_like(w0) + 1e-8

    w_hi = torch.clamp(2.0 * w0 + 1e-6, min=1e-6)

    for _ in range(8):
        p_hi = chi_square_independence_power(w_hi, sample_size, r, c, alpha)

        need_expand = p_hi < power
        if not torch.any(need_expand):
            break
        w_hi = torch.where(need_expand, w_hi * 2.0, w_hi)

        w_hi = torch.clamp(w_hi, max=torch.tensor(10.0, dtype=dtype))

    w = (w_lo + w_hi) * 0.5
    for _ in range(24):
        p_mid = chi_square_independence_power(w, sample_size, r, c, alpha)

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
