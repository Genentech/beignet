import math

import torch
from torch import Tensor

from ._mann_whitney_u_test_power import mann_whitney_u_test_power


def mann_whitney_u_test_sample_size(
    auc: Tensor,
    ratio: Tensor | float = 1.0,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Required group-1 sample size for Mann–Whitney U (normal approximation), parameterized by AUC.
    """
    auc = torch.atleast_1d(torch.as_tensor(auc))
    r = torch.as_tensor(ratio)
    dtype = (
        torch.float64
        if (auc.dtype == torch.float64 or r.dtype == torch.float64)
        else torch.float32
    )
    auc = auc.to(dtype)
    r = torch.clamp(r.to(dtype), min=0.1, max=10.0)

    # Initial z-approx for n1 using sd0 ≈ sqrt(n1*n2*(n1+n2+1)/12)
    sqrt2 = math.sqrt(2.0)

    def z_of(p: float) -> Tensor:
        pt = torch.tensor(p, dtype=dtype)
        eps = torch.finfo(dtype).eps
        pt = torch.clamp(pt, min=eps, max=1 - eps)
        return sqrt2 * torch.erfinv(2.0 * pt - 1.0)

    z_alpha = z_of(1 - (alpha / 2 if alternative == "two-sided" else alpha))
    z_beta = z_of(power)
    delta = torch.abs(auc - 0.5)
    # crude: n1 ≈ ((z_alpha+z_beta) * c / delta)^2, c from variance term
    c = torch.sqrt(1.0 + r) / torch.sqrt(12.0 * r)
    n1 = ((z_alpha + z_beta) * c / torch.clamp(delta, min=1e-8)) ** 2
    n1 = torch.clamp(n1, min=5.0)

    n1_curr = n1
    for _ in range(12):
        pwr = mann_whitney_u_test_power(
            auc, torch.ceil(n1_curr), ratio=r, alpha=alpha, alternative=alternative
        )
        gap = torch.clamp(power - pwr, min=-0.45, max=0.45)
        n1_curr = torch.clamp(n1_curr * (1.0 + 1.25 * gap), min=5.0, max=1e7)

    n1_out = torch.ceil(n1_curr)
    if out is not None:
        out.copy_(n1_out)
        return out
    return n1_out
