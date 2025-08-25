import math

import torch
from torch import Tensor


def welch_t_test_sample_size(
    effect_size: Tensor,
    ratio: Tensor | float = 1.0,
    var_ratio: Tensor | float = 1.0,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Required group-1 sample size for Welch's two-sample t-test.

    Solves for n1; n2 = ceil(n1 * ratio). Uses iterative refinement similar to
    other sample-size solvers with normal approximations.

    Parameters
    ----------
    effect_size : Tensor
        Standardized effect size d = (μ₁ − μ₂) / σ₁ using group 1 SD as unit.
        Should be > 0.

    ratio : Tensor or float, default=1.0
        Ratio n2/n1.

    var_ratio : Tensor or float, default=1.0
        Variance ratio σ₂²/σ₁².

    power : float, default=0.8
        Target power.

    alpha : float, default=0.05
        Significance level.

    alternative : str, default="two-sided"
        Alternative hypothesis ("two-sided", "greater", or "less").

    out : Tensor, optional
        Output tensor.

    Returns
    -------
    Tensor
        Required n1 (ceil). n2 = ceil(n1 * ratio).
    """
    d = torch.atleast_1d(torch.as_tensor(effect_size))
    r = torch.as_tensor(ratio)
    vr = torch.as_tensor(var_ratio)

    dtype = (
        torch.float64
        if any(t.dtype == torch.float64 for t in (d, r, vr))
        else torch.float32
    )
    d = d.to(dtype)
    r = r.to(dtype) if isinstance(r, Tensor) else torch.tensor(float(r), dtype=dtype)
    vr = (
        vr.to(dtype) if isinstance(vr, Tensor) else torch.tensor(float(vr), dtype=dtype)
    )

    d = torch.clamp(d, min=1e-8)
    r = torch.clamp(r, min=0.1, max=10.0)
    vr = torch.clamp(vr, min=1e-6, max=1e6)

    # Alternative normalization
    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    # Initial normal-based guess, treating df as large
    sqrt2 = math.sqrt(2.0)
    if alt == "two-sided":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt2
    else:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2

    # Welch SE with n1 unknown: se = sqrt(1/n1 + vr/(n1*ratio)) = sqrt((1 + vr/ratio)/n1)
    k = 1.0 + vr / r
    n1_guess = ((z_alpha + z_beta) * torch.sqrt(k) / d) ** 2
    n1_guess = torch.clamp(n1_guess, min=2.0)

    n1_curr = n1_guess
    max_iter = 12
    for _ in range(max_iter):
        n2_curr = torch.clamp(torch.ceil(n1_curr * r), min=2.0)
        # Welch SE and df
        a = 1.0 / n1_curr
        b = vr / n2_curr
        se2 = a + b
        se = torch.sqrt(se2)
        df = (se2**2) / (
            a**2 / torch.clamp(n1_curr - 1, min=1.0)
            + b**2 / torch.clamp(n2_curr - 1, min=1.0)
        )
        # Critical value with df adjustment
        if alt == "two-sided":
            tcrit = z_alpha * torch.sqrt(1 + 1 / (2 * df))
        else:
            tcrit = z_alpha * torch.sqrt(1 + 1 / (2 * df))
        # Noncentrality
        ncp = d / torch.clamp(se, min=1e-12)
        # Approx noncentral t variance
        var_nct = torch.where(
            df > 2,
            (df + ncp**2) / (df - 2),
            1 + ncp**2 / (2 * torch.clamp(df, min=1.0)),
        )
        std_nct = torch.sqrt(var_nct)
        if alt == "two-sided":
            zu = (tcrit - ncp) / torch.clamp(std_nct, min=1e-10)
            zl = (-tcrit - ncp) / torch.clamp(std_nct, min=1e-10)
            p_curr = 0.5 * (
                1 - torch.erf(zu / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
            ) + 0.5 * (1 + torch.erf(zl / torch.sqrt(torch.tensor(2.0, dtype=dtype))))
        elif alt == "greater":
            zscore = (tcrit - ncp) / torch.clamp(std_nct, min=1e-10)
            p_curr = 0.5 * (
                1 - torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
            )
        else:
            zscore = (-tcrit - ncp) / torch.clamp(std_nct, min=1e-10)
            p_curr = 0.5 * (
                1 + torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
            )

        # Update n1 heuristically based on power gap
        gap = torch.clamp(power - p_curr, min=-0.49, max=0.49)
        n1_curr = torch.clamp(n1_curr * (1.0 + 1.25 * gap), min=2.0, max=1e7)

    result = torch.ceil(n1_curr)
    result = torch.clamp(result, min=2.0)
    if out is not None:
        out.copy_(result)
        return out
    return result
