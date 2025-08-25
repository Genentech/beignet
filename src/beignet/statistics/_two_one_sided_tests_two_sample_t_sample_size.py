import math

import torch
from torch import Tensor

from ._two_one_sided_tests_two_sample_t_power import (
    two_one_sided_tests_two_sample_t_power,
)


def two_one_sided_tests_two_sample_t_sample_size(
    true_effect: Tensor,
    ratio: Tensor | float = 1.0,
    low: Tensor = 0.0,
    high: Tensor = 0.0,
    power: float = 0.8,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Required group-1 sample size for two-sample Two One-Sided Tests (equivalence).

    Solves for n1; n2 = ceil(n1 * ratio). Uses iterative refinement with the
    same approximations as the power function.
    """
    d = torch.atleast_1d(torch.as_tensor(true_effect))
    r = torch.as_tensor(ratio)
    low = torch.atleast_1d(torch.as_tensor(low))
    high = torch.atleast_1d(torch.as_tensor(high))

    dtype = (
        torch.float64
        if any(t.dtype == torch.float64 for t in (d, r, low, high))
        else torch.float32
    )
    d = d.to(dtype)
    low = low.to(dtype)
    high = high.to(dtype)
    r = r.to(dtype) if isinstance(r, Tensor) else torch.tensor(float(r), dtype=dtype)
    r = torch.clamp(r, min=0.1, max=10.0)

    # Initial guess (z-approx with pooled SE factor)
    sqrt2 = math.sqrt(2.0)
    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2
    margin = torch.minimum(d - low, high - d)
    margin = torch.clamp(margin, min=1e-8)
    # se_factor ≈ sqrt(1/n1 + 1/(n1*ratio)) => sqrt((1+1/ratio)/n1)
    c = torch.sqrt(1.0 + 1.0 / r)
    n1 = ((z_alpha + z_beta) * c / margin) ** 2
    n1 = torch.clamp(n1, min=2.0)

    n1_curr = n1
    for _ in range(12):
        p = two_one_sided_tests_two_sample_t_power(
            d, n1_curr, ratio=r, low=low, high=high, alpha=alpha
        )
        gap = torch.clamp(power - p, min=-0.45, max=0.45)
        n1_curr = torch.clamp(n1_curr * (1.0 + 1.25 * gap), min=2.0, max=1e7)

    n1_out = torch.ceil(n1_curr)
    if out is not None:
        out.copy_(n1_out)
        return out
    return n1_out
