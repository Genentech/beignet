import math

import torch
from torch import Tensor

from ._wilcoxon_signed_rank_test_power import wilcoxon_signed_rank_test_power


def wilcoxon_signed_rank_test_minimum_detectable_effect(
    nobs: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Minimum detectable prob_positive = P(D>0) + 0.5 P(D=0) for Wilcoxon signed-rank test.

    Parameters
    ----------
    nobs : Tensor
        Number of non-zero paired differences.
    power : float, default=0.8
        Target power.
    alpha : float, default=0.05
        Significance level.
    alternative : {"two-sided", "greater", "less"}
        Direction; returns ≥0.5 for "greater"/"two-sided" and ≤0.5 for "less".

    Returns
    -------
    Tensor
        Minimal detectable prob_positive value.
    """
    n0 = torch.as_tensor(nobs)
    scalar_out = n0.ndim == 0
    n = torch.atleast_1d(n0)
    dtype = torch.float64 if n.dtype == torch.float64 else torch.float32
    n = torch.clamp(n.to(dtype), min=5.0)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    # H0 moments
    S = n * (n + 1.0) / 2.0
    var0 = n * (n + 1.0) * (2.0 * n + 1.0) / 24.0
    sd0 = torch.sqrt(torch.clamp(var0, min=1e-12))

    sqrt2 = math.sqrt(2.0)

    def z_of(p: float) -> Tensor:
        q = torch.tensor(p, dtype=dtype)
        eps = torch.finfo(dtype).eps
        q = torch.clamp(q, min=eps, max=1 - eps)
        return sqrt2 * torch.erfinv(2.0 * q - 1.0)

    z_alpha = z_of(1 - alpha / 2) if alt == "two-sided" else z_of(1 - alpha)
    z_beta = z_of(power)
    delta = (z_alpha + z_beta) * sd0 / torch.clamp(S, min=1e-12)

    if alt == "less":
        p = torch.clamp(0.5 - delta, 0.0, 1.0)
    else:
        p = torch.clamp(0.5 + delta, 0.0, 1.0)

    # Optional small refinement
    p_curr = wilcoxon_signed_rank_test_power(p, n, alpha=alpha, alternative=alt)
    gap = torch.clamp(power - p_curr, min=-0.45, max=0.45)
    step = gap * 0.02
    if alt == "less":
        p = torch.clamp(p - torch.abs(step), 0.0, 1.0)
    else:
        p = torch.clamp(p + torch.abs(step), 0.0, 1.0)

    if scalar_out:
        p_s = p.reshape(())
        if out is not None:
            out.copy_(p_s)
            return out
        return p_s
    else:
        if out is not None:
            out.copy_(p)
            return out
        return p
