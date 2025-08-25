import math

import torch
from torch import Tensor


def paired_t_test_power(
    effect_size: Tensor,
    sample_size: Tensor,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Power for paired-samples t-test (unknown variance of differences).

    Parameters
    ----------
    effect_size : Tensor
        Standardized mean difference of pairs d = μ_d/σ_d.
    sample_size : Tensor
        Number of pairs (n >= 2).
    alpha : float, default=0.05
        Significance level.
    alternative : {"two-sided", "greater", "less"}
        Alternative hypothesis.

    Returns
    -------
    Tensor
        Statistical power.
    """
    d = torch.atleast_1d(torch.as_tensor(effect_size))
    n = torch.atleast_1d(torch.as_tensor(sample_size))
    dtype = (
        torch.float64
        if (d.dtype == torch.float64 or n.dtype == torch.float64)
        else torch.float32
    )
    d = d.to(dtype)
    n = torch.clamp(n.to(dtype), min=2.0)

    df = n - 1
    ncp = d * torch.sqrt(n)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    sqrt2 = math.sqrt(2.0)
    # Critical value (normal-based with finite-df adjustment)
    if alt == "two-sided":
        z = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt2
        tcrit = z * torch.sqrt(1 + 1 / (2 * df))
    else:
        z = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
        tcrit = z * torch.sqrt(1 + 1 / (2 * df))

    # Approximate noncentral t by normal with mean=ncp and var=(df+ncp^2)/(df-2)
    var_nct = torch.where(
        df > 2, (df + ncp**2) / (df - 2), 1 + ncp**2 / (2 * torch.clamp(df, min=1.0))
    )
    std_nct = torch.sqrt(var_nct)

    if alt == "two-sided":
        zu = (tcrit - ncp) / torch.clamp(std_nct, min=1e-10)
        zl = (-tcrit - ncp) / torch.clamp(std_nct, min=1e-10)
        power = 0.5 * (
            1 - torch.erf(zu / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
        ) + 0.5 * (1 + torch.erf(zl / torch.sqrt(torch.tensor(2.0, dtype=dtype))))
    elif alt == "greater":
        zscore = (tcrit - ncp) / torch.clamp(std_nct, min=1e-10)
        power = 0.5 * (
            1 - torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
        )
    else:
        zscore = (-tcrit - ncp) / torch.clamp(std_nct, min=1e-10)
        power = 0.5 * (
            1 + torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
        )

    out_t = torch.clamp(power, 0.0, 1.0)
    if out is not None:
        out.copy_(out_t)
        return out
    return out_t
