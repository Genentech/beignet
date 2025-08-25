import math

import torch
from torch import Tensor


def wilcoxon_signed_rank_test_power(
    prob_positive: Tensor,
    nobs: Tensor,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Power for Wilcoxon signed-rank test (normal approximation).

    Parameters
    ----------
    prob_positive : Tensor
        Probability that a paired difference is positive, i.e., P(D > 0) + 0.5 P(D = 0).
        Under the null, this equals 0.5.
    nobs : Tensor
        Number of paired observations (after removing zero differences).
    alpha : float, default=0.05
        Significance level.
    alternative : {"two-sided", "greater", "less"}
        Test direction. "greater" tests for median(D) > 0, "less" for < 0.

    Returns
    -------
    Tensor
        Statistical power.

    Notes
    -----
    Uses the large-sample normal approximation for the sum of positive ranks W⁺.
    Under H0: E[W⁺] = S/2, Var[W⁺] = n(n+1)(2n+1)/24, where S = n(n+1)/2.
    We approximate E[W⁺|H1] ≈ S * prob_positive and use Var[W⁺] from H0.
    This parallels the Mann–Whitney implementation parameterized by AUC.
    """
    p = torch.atleast_1d(torch.as_tensor(prob_positive))
    n = torch.atleast_1d(torch.as_tensor(nobs))

    dtype = (
        torch.float64
        if (p.dtype == torch.float64 or n.dtype == torch.float64)
        else torch.float32
    )
    p = p.to(dtype)
    n = torch.clamp(n.to(dtype), min=5.0)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    # Sum of ranks S and H0 moments for W+
    S = n * (n + 1.0) / 2.0
    mean0 = S / 2.0
    var0 = n * (n + 1.0) * (2.0 * n + 1.0) / 24.0
    sd0 = torch.sqrt(torch.clamp(var0, min=1e-12))

    # H1 mean using prob_positive; variance approx by var0
    mean1 = S * p
    ncp = (mean1 - mean0) / sd0

    sqrt2 = math.sqrt(2.0)

    def z_of(prob: float) -> Tensor:
        q = torch.tensor(prob, dtype=dtype)
        eps = torch.finfo(dtype).eps
        q = torch.clamp(q, min=eps, max=1 - eps)
        return sqrt2 * torch.erfinv(2.0 * q - 1.0)

    if alt == "two-sided":
        zcrit = z_of(1 - alpha / 2)
        upper = 0.5 * (1 - torch.erf((zcrit - ncp) / sqrt2))
        lower = 0.5 * (1 + torch.erf((-zcrit - ncp) / sqrt2))
        power = upper + lower
    elif alt == "greater":
        zcrit = z_of(1 - alpha)
        power = 0.5 * (1 - torch.erf((zcrit - ncp) / sqrt2))
    else:
        zcrit = z_of(1 - alpha)
        power = 0.5 * (1 + torch.erf((-zcrit - ncp) / sqrt2))

    out_t = torch.clamp(power, 0.0, 1.0)
    if out is not None:
        out.copy_(out_t)
        return out
    return out_t
