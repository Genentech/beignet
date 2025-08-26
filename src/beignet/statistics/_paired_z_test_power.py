import math

import torch
from torch import Tensor


def paired_z_test_power(
    effect_size: Tensor,
    sample_size: Tensor,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Power for paired-samples z-test (known variance of differences).

    Parameters
    ----------
    effect_size : Tensor
        Standardized mean difference of pairs d = μ_d/σ_d.
    sample_size : Tensor
        Number of pairs.
    alpha : float, default=0.05
        Significance level.
    alternative : {"two-sided", "greater", "less"}
        Alternative hypothesis.

    Returns
    -------
    Tensor
        Statistical power.

    When to Use
    -----------
    **Traditional Statistics:**

    - **Large sample paired comparisons:** When n > 30 and population variance is known
    - **Before-after studies with known variance:** Pre/post intervention comparisons
    - **Repeated measures analysis:** When within-subject variance is known from prior studies
    - **Clinical trial planning:** Large-scale paired studies with established variance parameters
    - **Quality control monitoring:** Paired measurements with known measurement error
    - **Educational assessment:** Pre/post test comparisons with known test reliability

    **Machine Learning Applications:**

    - **Large-scale A/B testing:** User-level paired comparisons with known variance from historical data
    - **Model performance evaluation:** Paired testing when validation variance is well-established
    - **Algorithm benchmarking:** Large sample comparisons with established performance variance
    - **Production monitoring:** Paired system performance metrics with known baseline variance
    - **Recommendation system evaluation:** User engagement comparisons with historical variance
    - **Search algorithm testing:** Query performance improvements with established variance
    - **Content optimization:** Paired content performance with known engagement variance
    - **Ad targeting effectiveness:** Campaign performance with established CTR variance
    - **Fraud detection validation:** False positive/negative rate comparisons with known variance
    - **Customer behavior analysis:** Purchase pattern changes with established behavioral variance
    - **Personalization impact:** Individual user response improvements with known variance
    - **Load testing evaluation:** System performance under different conditions with known variance
    - **Feature deployment analysis:** Performance impact with established baseline variance
    - **Conversion rate optimization:** User journey improvements with historical conversion variance
    - **Churn prediction validation:** Model performance improvements with established churn variance

    **Interpretation Guidelines:**

    - **Effect size = 0.2:** Small effect (Cohen's convention)
    - **Effect size = 0.5:** Medium effect
    - **Effect size = 0.8:** Large effect
    - **Use when σ is known:** Z-test assumes known population variance
    - **Large sample assumption:** Generally appropriate when n > 30
    - **Consider practical significance:** Statistical power should align with meaningful effect sizes
    """
    d = torch.atleast_1d(torch.as_tensor(effect_size))
    n = torch.atleast_1d(torch.as_tensor(sample_size))
    dtype = (
        torch.float64
        if (d.dtype == torch.float64 or n.dtype == torch.float64)
        else torch.float32
    )
    d = d.to(dtype)
    n = torch.clamp(n.to(dtype), min=1.0)

    ncp = d * torch.sqrt(n)
    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    sqrt2 = math.sqrt(2.0)

    def z_of(p: float) -> torch.Tensor:
        pt = torch.tensor(p, dtype=dtype)
        eps = torch.finfo(dtype).eps
        pt = torch.clamp(pt, min=eps, max=1 - eps)
        return sqrt2 * torch.erfinv(2.0 * pt - 1.0)

    if alt == "two-sided":
        zcrit = z_of(1 - alpha / 2)
        upper = 0.5 * (
            1 - torch.erf((zcrit - ncp) / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
        )
        lower = 0.5 * (
            1 + torch.erf((-zcrit - ncp) / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
        )
        power = upper + lower
    elif alt == "greater":
        zcrit = z_of(1 - alpha)
        power = 0.5 * (
            1 - torch.erf((zcrit - ncp) / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
        )
    else:
        zcrit = z_of(1 - alpha)
        power = 0.5 * (
            1 + torch.erf((-zcrit - ncp) / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
        )

    out_t = torch.clamp(power, 0.0, 1.0)
    if out is not None:
        out.copy_(out_t)
        return out
    return out_t
