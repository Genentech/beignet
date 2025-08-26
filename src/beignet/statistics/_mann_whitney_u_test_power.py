import math

import torch
from torch import Tensor


def mann_whitney_u_test_power(
    auc: Tensor,
    nobs1: Tensor,
    nobs2: Tensor | None = None,
    ratio: Tensor | float = 1.0,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Power for Mann–Whitney U test (normal approximation), parameterized by AUC.

    When to Use
    -----------
    **Traditional Statistics:**
    - Comparing two independent groups when normality assumptions are violated
    - Ordinal or ranked data comparisons
    - Small sample sizes where t-test assumptions fail
    - Robust alternative to independent t-test

    **Machine Learning Contexts:**
    - Comparing model performance between two algorithms (non-parametric)
    - A/B testing with non-normal metrics (conversion rates, engagement times)
    - Fairness assessment: comparing outcomes between demographic groups
    - Feature importance: comparing feature distributions across outcome classes
    - Model validation: comparing prediction quality across different data subsets
    - Anomaly detection: comparing normal vs. anomalous data distributions
    - Time series: comparing performance across different time periods
    - Transfer learning: validating domain adaptation effectiveness

    **Choose Mann-Whitney U over t-test when:**
    - Data is not normally distributed
    - Outcome variable is ordinal
    - Presence of outliers that would affect parametric tests
    - Small sample sizes (n < 30 per group)
    - Heterogeneous variances between groups

    **AUC Interpretation:**
    - AUC = 0.5: no difference between groups
    - AUC = 0.56: small effect (Cohen's d ≈ 0.2)
    - AUC = 0.64: medium effect (Cohen's d ≈ 0.5)
    - AUC = 0.71: large effect (Cohen's d ≈ 0.8)

    Parameters
    ----------
    auc : Tensor
        P(X > Y) + 0.5 P(X=Y). Null is 0.5.
    nobs1 : Tensor
        Sample size group 1.
    nobs2 : Tensor, optional
        Sample size group 2. If None, uses ratio * nobs1.
    ratio : Tensor or float, default=1.0
        nobs2/nobs1 when nobs2 is None.
    alpha : float, default=0.05
    alternative : {"two-sided", "greater", "less"}

    Returns
    -------
    Tensor
        Statistical power.
    """
    auc = torch.atleast_1d(torch.as_tensor(auc))
    sample_size_group_1 = torch.atleast_1d(torch.as_tensor(nobs1))
    if nobs2 is None:
        r = torch.as_tensor(ratio)
        sample_size_group_2 = torch.ceil(
            sample_size_group_1
            * (
                r.to(sample_size_group_1.dtype)
                if isinstance(r, Tensor)
                else torch.tensor(float(r), dtype=sample_size_group_1.dtype)
            )
        )
    else:
        sample_size_group_2 = torch.atleast_1d(torch.as_tensor(nobs2))

    dtype = (
        torch.float64
        if any(
            t.dtype == torch.float64
            for t in (auc, sample_size_group_1, sample_size_group_2)
        )
        else torch.float32
    )
    auc = auc.to(dtype)
    sample_size_group_1 = torch.clamp(sample_size_group_1.to(dtype), min=2.0)
    sample_size_group_2 = torch.clamp(sample_size_group_2.to(dtype), min=2.0)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    # Normal approx for U: under H0, mean0 = sample_size_group_1*sample_size_group_2/2, var0 = sample_size_group_1*sample_size_group_2*(sample_size_group_1+sample_size_group_2+1)/12
    # Under H1, mean1 = sample_size_group_1*sample_size_group_2*auc, use var0 as approximation
    mean0 = sample_size_group_1 * sample_size_group_2 / 2.0
    var0 = (
        sample_size_group_1
        * sample_size_group_2
        * (sample_size_group_1 + sample_size_group_2 + 1.0)
        / 12.0
    )
    mean1 = sample_size_group_1 * sample_size_group_2 * auc

    # Z = (U - mean0)/sd0; under H1, Z ~ N((mean1-mean0)/sd0, 1)
    sd0 = torch.sqrt(torch.clamp(var0, min=1e-12))
    noncentrality_parameter = (mean1 - mean0) / sd0

    sqrt2 = math.sqrt(2.0)

    def z_of(p: float) -> Tensor:
        pt = torch.tensor(p, dtype=dtype)
        eps = torch.finfo(dtype).eps
        pt = torch.clamp(pt, min=eps, max=1 - eps)
        return sqrt2 * torch.erfinv(2.0 * pt - 1.0)

    if alt == "two-sided":
        zcrit = z_of(1 - alpha / 2)
        # Power: P(|Z|>zcrit) with Z~N(noncentrality_parameter,1)
        upper = 0.5 * (
            1 - torch.erf((zcrit - noncentrality_parameter) / math.sqrt(2.0))
        )
        lower = 0.5 * (
            1 + torch.erf((-zcrit - noncentrality_parameter) / math.sqrt(2.0))
        )
        power = upper + lower
    elif alt == "greater":
        zcrit = z_of(1 - alpha)
        power = 0.5 * (
            1 - torch.erf((zcrit - noncentrality_parameter) / math.sqrt(2.0))
        )
    else:
        zcrit = z_of(1 - alpha)
        power = 0.5 * (
            1 + torch.erf((-zcrit - noncentrality_parameter) / math.sqrt(2.0))
        )

    out_t = torch.clamp(power, 0.0, 1.0)
    if out is not None:
        out.copy_(out_t)
        return out
    return out_t
