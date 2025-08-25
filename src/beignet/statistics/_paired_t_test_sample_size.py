import math

import torch
from torch import Tensor


def paired_t_test_sample_size(
    effect_size: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Required sample size (number of pairs) for paired t-test.

    Parameters
    ----------
    effect_size : Tensor
        Standardized mean difference of pairs d = μ_d/σ_d. (d > 0).
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

    if alt == "two-sided":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt2
    else:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2

    # Initial normal approximation: n = ((z_alpha + z_beta)/d)^2
    n = ((z_alpha + z_beta) / d) ** 2
    n = torch.clamp(n, min=2.0)

    # Iterative df correction similar to t_test_sample_size
    n_curr = n
    for _ in range(10):
        df = torch.clamp(n_curr - 1, min=1.0)
        tcrit = z_alpha * torch.sqrt(1 + 1 / (2 * df))
        ncp = d * torch.sqrt(n_curr)
        var_nct = torch.where(
            df > 2,
            (df + ncp**2) / (df - 2),
            1 + ncp**2 / (2 * torch.clamp(df, min=1.0)),
        )
        std_nct = torch.sqrt(var_nct)
        if alt == "two-sided":
            zu = (tcrit - ncp) / torch.clamp(std_nct, min=1e-10)
            zl = (-tcrit - ncp) / torch.clamp(std_nct, min=1e-10)
            p = 0.5 * (
                1 - torch.erf(zu / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
            ) + 0.5 * (1 + torch.erf(zl / torch.sqrt(torch.tensor(2.0, dtype=dtype))))
        elif alt == "greater":
            zscore = (tcrit - ncp) / torch.clamp(std_nct, min=1e-10)
            p = 0.5 * (
                1 - torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
            )
        else:
            zscore = (-tcrit - ncp) / torch.clamp(std_nct, min=1e-10)
            p = 0.5 * (
                1 + torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
            )
        gap = torch.clamp(power - p, min=-0.45, max=0.45)
        n_curr = torch.clamp(n_curr * (1.0 + 1.25 * gap), min=2.0, max=1e7)

    n_out = torch.ceil(n_curr)
    if out is not None:
        out.copy_(n_out)
        return out
    return n_out
