import math

import torch
from torch import Tensor


def welch_t_test_power(
    effect_size: Tensor,
    nobs1: Tensor,
    nobs2: Tensor,
    var_ratio: Tensor | float = 1.0,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute statistical power for Welch's two-sample t-test.

    Welch's t-test allows unequal variances and sample sizes.

    Parameters
    ----------
    effect_size : Tensor
        Standardized effect size d = (μ₁ − μ₂) / σ₁ using group 1 SD as unit.
        Should be non-negative for power (direction handled by `alternative`).

    nobs1 : Tensor
        Sample size for group 1.

    nobs2 : Tensor
        Sample size for group 2.

    var_ratio : Tensor or float, default=1.0
        Variance ratio σ₂²/σ₁². Use 1.0 for equal variances.

    alpha : float, default=0.05
        Significance level (Type I error rate).

    alternative : str, default="two-sided"
        Type of alternative hypothesis: "two-sided", "greater", or "less".

    out : Tensor, optional
        Output tensor. Default, `None`.

    Returns
    -------
    output : Tensor
        Statistical power (probability of correctly rejecting false null hypothesis).
    """
    d = torch.atleast_1d(torch.as_tensor(effect_size))
    n1 = torch.atleast_1d(torch.as_tensor(nobs1))
    n2 = torch.atleast_1d(torch.as_tensor(nobs2))
    vr = torch.as_tensor(var_ratio)

    # Dtype
    if any(
        t.dtype == torch.float64
        for t in (d, n1, n2, vr if isinstance(vr, Tensor) else torch.tensor(0.0))
    ):
        dtype = torch.float64
    else:
        dtype = torch.float32
    d = d.to(dtype)
    n1 = n1.to(dtype)
    n2 = n2.to(dtype)
    if isinstance(vr, Tensor):
        vr = vr.to(dtype)
    else:
        vr = torch.tensor(float(vr), dtype=dtype)

    # Clamp
    d = torch.clamp(d, min=0.0)
    n1 = torch.clamp(n1, min=2.0)
    n2 = torch.clamp(n2, min=2.0)
    vr = torch.clamp(vr, min=1e-6, max=1e6)

    # Welch SE and df
    a = 1.0 / n1
    b = vr / n2
    se2 = a + b
    se = torch.sqrt(se2)
    df = (se2**2) / (
        a**2 / torch.clamp(n1 - 1, min=1.0) + b**2 / torch.clamp(n2 - 1, min=1.0)
    )

    # Noncentrality parameter
    ncp = d / torch.clamp(se, min=1e-12)

    # Alternative normalization
    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    # Critical value approx (normal with df adjustment)
    sqrt2 = math.sqrt(2.0)
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
    else:  # less
        zscore = (-tcrit - ncp) / torch.clamp(std_nct, min=1e-10)
        power = 0.5 * (
            1 + torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
        )

    output = torch.clamp(power, 0.0, 1.0)
    if out is not None:
        out.copy_(output)
        return out
    return output
