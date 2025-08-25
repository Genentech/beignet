import math

import torch
from torch import Tensor

from ._wilcoxon_signed_rank_test_power import wilcoxon_signed_rank_test_power


def wilcoxon_signed_rank_test_sample_size(
    prob_positive: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Required sample size (pairs) for Wilcoxon signed-rank test (normal approximation).

    Parameters
    ----------
    prob_positive : Tensor
        Probability that a paired difference is positive, i.e., P(D > 0) + 0.5 P(D = 0).
        Null is 0.5. Values farther from 0.5 indicate stronger effects.
    power : float, default=0.8
        Desired power.
    alpha : float, default=0.05
        Significance level.
    alternative : {"two-sided", "greater", "less"}
        Test direction.

    Returns
    -------
    Tensor
        Required number of non-zero pairs (rounded up).
    """
    p = torch.atleast_1d(torch.as_tensor(prob_positive))
    dtype = torch.float64 if p.dtype == torch.float64 else torch.float32
    p = p.to(dtype)

    sqrt3 = math.sqrt(3.0)
    sqrt2 = math.sqrt(2.0)

    def z_of(prob: float) -> Tensor:
        q = torch.tensor(prob, dtype=dtype)
        eps = torch.finfo(dtype).eps
        q = torch.clamp(q, min=eps, max=1 - eps)
        return sqrt2 * torch.erfinv(2.0 * q - 1.0)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        z_alpha = z_of(1 - alpha)
    elif alt in {"smaller", "less", "<"}:
        z_alpha = z_of(1 - alpha)
    elif alt == "two-sided":
        z_alpha = z_of(1 - alpha / 2)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    z_beta = z_of(power)
    delta = torch.abs(p - 0.5)

    # Asymptotic initial guess: ncp ≈ sqrt(3) * sqrt(n) * (p - 0.5)
    # Solve for n with target (z_alpha + z_beta)
    n0 = ((z_alpha + z_beta) / (sqrt3 * torch.clamp(delta, min=1e-8))) ** 2
    n0 = torch.clamp(n0, min=5.0)

    n_curr = n0
    for _ in range(12):
        pwr = wilcoxon_signed_rank_test_power(
            p, torch.ceil(n_curr), alpha=alpha, alternative=alt
        )
        gap = torch.clamp(power - pwr, min=-0.45, max=0.45)
        n_curr = torch.clamp(n_curr * (1.0 + 1.25 * gap), min=5.0, max=1e7)

    n_out = torch.ceil(n_curr)
    if out is not None:
        out.copy_(n_out)
        return out
    return n_out
