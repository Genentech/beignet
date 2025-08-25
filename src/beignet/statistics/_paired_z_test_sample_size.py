import math

import torch
from torch import Tensor


def paired_z_test_sample_size(
    effect_size: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Required sample size for paired z-test (known variance of differences).

    Parameters
    ----------
    effect_size : Tensor
        Standardized mean difference of pairs d = μ_d/σ_d. Should be > 0.
    power : float, default=0.8
        Target power.
    alpha : float, default=0.05
        Significance level.
    alternative : {"two-sided", "greater", "less"}
        Alternative hypothesis.

    Returns
    -------
    Tensor
        Required number of pairs (ceil).
    """
    d = torch.atleast_1d(torch.as_tensor(effect_size))
    dtype = torch.float64 if d.dtype == torch.float64 else torch.float32
    d = torch.clamp(d.to(dtype), min=1e-8)

    sqrt2 = math.sqrt(2.0)
    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    def z_of(p: float) -> torch.Tensor:
        pt = torch.tensor(p, dtype=dtype)
        eps = torch.finfo(dtype).eps
        pt = torch.clamp(pt, min=eps, max=1 - eps)
        return sqrt2 * torch.erfinv(2.0 * pt - 1.0)

    if alt == "two-sided":
        z_alpha = z_of(1 - alpha / 2)
    else:
        z_alpha = z_of(1 - alpha)
    z_beta = z_of(power)

    n = ((z_alpha + z_beta) / d) ** 2
    n = torch.clamp(n, min=1.0)
    n = torch.ceil(n)
    if out is not None:
        out.copy_(n)
        return out
    return n
