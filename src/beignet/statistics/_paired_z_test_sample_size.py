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

    When to Use
    -----------
    **Traditional Statistics:**

    - **Large sample paired studies:** Planning studies when population variance is known
    - **Clinical trial sample size planning:** Paired designs with established variance parameters
    - **Quality control studies:** Sample size for paired measurement comparisons
    - **Educational research planning:** Pre/post test studies with known test reliability
    - **Longitudinal study design:** Planning repeated measures with known within-subject variance
    - **Laboratory method comparison:** Sample size for paired analytical method validation

    **Machine Learning Applications:**

    - **A/B testing sample size planning:** User-level paired experiments with historical variance
    - **Model evaluation study design:** Planning paired performance comparisons with known variance
    - **Algorithm benchmarking studies:** Sample size for paired algorithm comparisons
    - **Production experiment planning:** System performance studies with established variance
    - **Recommendation system testing:** Sample size for paired user engagement studies
    - **Feature impact studies:** Planning paired feature evaluation with known performance variance
    - **Conversion optimization planning:** Sample size for paired user journey comparisons
    - **Search algorithm evaluation:** Planning paired query performance studies
    - **Personalization effectiveness studies:** Sample size for individual user response analysis
    - **Fraud detection validation:** Planning paired detection rate comparisons
    - **Load testing experiments:** Sample size for paired system performance evaluation
    - **Churn prediction studies:** Planning model performance comparisons with known variance
    - **Content effectiveness testing:** Sample size for paired content performance evaluation
    - **User experience optimization:** Planning paired UX metric comparisons
    - **Machine learning pipeline validation:** Sample size for paired processing method comparisons

    **Interpretation Guidelines:**

    - **Effect size = 0.2:** Small effect requires larger samples (n ≈ 393 for 80% power)
    - **Effect size = 0.5:** Medium effect (n ≈ 63 for 80% power)
    - **Effect size = 0.8:** Large effect (n ≈ 25 for 80% power)
    - **Use when variance is known:** Z-test assumes known population variance of differences
    - **Consider practical constraints:** Balance statistical requirements with resource limitations
    - **Account for dropouts:** Inflate sample size for expected attrition in longitudinal studies
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    dtype = torch.float64 if effect_size.dtype == torch.float64 else torch.float32
    effect_size = torch.clamp(effect_size.to(dtype), min=1e-8)

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

    sample_size = ((z_alpha + z_beta) / effect_size) ** 2
    sample_size = torch.clamp(sample_size, min=1.0)
    sample_size = torch.ceil(sample_size)
    if out is not None:
        out.copy_(sample_size)
        return out
    return sample_size
