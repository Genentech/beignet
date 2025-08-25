import math

import torch
from torch import Tensor


def mcnemars_test_power(
    p01: Tensor,
    p10: Tensor,
    sample_size: Tensor,
    alpha: float = 0.05,
    two_sided: bool = True,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Power for McNemar's test using a normal approximation.

    Parameters
    ----------
    p01 : Tensor
        Probability of discordant outcome (0→1).
    p10 : Tensor
        Probability of discordant outcome (1→0).
    sample_size : Tensor
        Number of paired observations.
    alpha : float, default=0.05
        Significance level.
    two_sided : bool, default=True
        Whether to use a two-sided test.

    Returns
    -------
    Tensor
        Statistical power.
    """
    p01 = torch.atleast_1d(torch.as_tensor(p01))
    p10 = torch.atleast_1d(torch.as_tensor(p10))
    n = torch.atleast_1d(torch.as_tensor(sample_size))
    dtype = (
        torch.float64
        if any(t.dtype == torch.float64 for t in (p01, p10, n))
        else torch.float32
    )
    p01 = torch.clamp(p01.to(dtype), 0.0, 1.0)
    p10 = torch.clamp(p10.to(dtype), 0.0, 1.0)
    n = torch.clamp(n.to(dtype), min=1.0)

    # Approximate using normal test on b - c with D ≈ n*(p01+p10)
    D = n * (p01 + p10)
    p = torch.where(
        (p01 + p10) > 0, p01 / torch.clamp(p01 + p10, min=1e-12), torch.zeros_like(p01)
    )
    # Under H1, E[b - D/2] = D*(p - 0.5); Var ≈ D*0.25
    mean = D * (p - 0.5)
    std = torch.sqrt(torch.clamp(D * 0.25, min=1e-12))

    sqrt2 = math.sqrt(2.0)

    def z_of(prob: float) -> Tensor:
        pt = torch.tensor(prob, dtype=dtype)
        eps = torch.finfo(dtype).eps
        pt = torch.clamp(pt, min=eps, max=1 - eps)
        return sqrt2 * torch.erfinv(2.0 * pt - 1.0)

    if two_sided:
        zcrit = z_of(1 - alpha / 2)
        # Power ≈ P(|Z| > zcrit) where Z ~ N(mean/std, 1)
        z_upper = zcrit - mean / torch.clamp(std, min=1e-12)
        z_lower = -zcrit - mean / torch.clamp(std, min=1e-12)
        power = 0.5 * (1 - torch.erf(z_upper / math.sqrt(2.0))) + 0.5 * (
            1 + torch.erf(z_lower / math.sqrt(2.0))
        )
    else:
        zcrit = z_of(1 - alpha)
        zscore = zcrit - mean / torch.clamp(std, min=1e-12)
        power = 0.5 * (1 - torch.erf(zscore / math.sqrt(2.0)))

    out_t = torch.clamp(power, 0.0, 1.0)
    if out is not None:
        out.copy_(out_t)
        return out
    return out_t
