import math

import torch
from torch import Tensor

from ._mann_whitney_u_test_power import mann_whitney_u_test_power


def mann_whitney_u_test_sample_size(
    auc: Tensor,
    ratio: Tensor | float = 1.0,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Required group-1 sample size for Mann–Whitney U (normal approximation), parameterized by AUC.

    When to Use
    -----------
    **Traditional Statistics:**

    - **Non-parametric study planning:** When distributions are expected to be non-normal
    - **Ordinal data analysis:** Planning studies with ranked or ordered categorical data
    - **Robust comparison studies:** When outliers are expected and normality can't be assumed
    - **Small sample planning:** Alternative to t-test when normality assumptions are questionable
    - **Skewed data studies:** Planning comparisons when data distributions are asymmetric
    - **Distribution-free testing:** When no assumptions about population distributions are desired

    **Machine Learning Applications:**

    - **Binary classification planning:** Sample size for ROC AUC-based model comparison
    - **Ranking algorithm evaluation:** Planning studies for ranking quality assessment
    - **Recommendation system testing:** Sample size for recommendation performance studies
    - **Anomaly detection validation:** Planning discrimination studies between normal/abnormal cases
    - **Information retrieval planning:** Sample size for search result quality comparisons
    - **Feature importance studies:** Planning non-parametric feature discrimination analysis
    - **Model fairness evaluation:** Sample size for bias detection across demographic groups
    - **A/B testing with ordinal outcomes:** Planning experiments with ranked response metrics
    - **Clustering validation planning:** Sample size for non-parametric cluster separation studies
    - **Time series anomaly detection:** Planning temporal pattern discrimination studies
    - **Multi-class classification:** Sample size for pairwise class discrimination analysis
    - **Ensemble method evaluation:** Planning non-parametric ensemble component comparison
    - **Transfer learning studies:** Sample size for domain adaptation discrimination analysis
    - **Active learning planning:** Sample size for uncertainty-based sample selection studies
    - **Personalization effectiveness:** Planning individual response discrimination studies

    **Interpretation Guidelines:**

    - **AUC = 0.6:** Small effect, requires large samples (n ≈ 863 per group for 80% power)
    - **AUC = 0.7:** Medium effect (n ≈ 199 per group for 80% power)
    - **AUC = 0.8:** Large effect (n ≈ 86 per group for 80% power)
    - **AUC = 0.9:** Very large effect (n ≈ 34 per group for 80% power)
    - **Distribution-free approach:** Valid regardless of underlying population distributions
    - **Consider practical significance:** AUC thresholds should be meaningful in application context
    """
    auc = torch.atleast_1d(torch.as_tensor(auc))
    r = torch.as_tensor(ratio)
    dtype = (
        torch.float64
        if (auc.dtype == torch.float64 or r.dtype == torch.float64)
        else torch.float32
    )
    auc = auc.to(dtype)
    r = torch.clamp(r.to(dtype), min=0.1, max=10.0)

    # Initial z-approx for sample_size_group_1 using sd0 ≈ sqrt(sample_size_group_1*sample_size_group_2*(sample_size_group_1+sample_size_group_2+1)/12)
    sqrt2 = math.sqrt(2.0)

    def z_of(p: float) -> Tensor:
        pt = torch.tensor(p, dtype=dtype)
        eps = torch.finfo(dtype).eps
        pt = torch.clamp(pt, min=eps, max=1 - eps)
        return sqrt2 * torch.erfinv(2.0 * pt - 1.0)

    z_alpha = z_of(1 - (alpha / 2 if alternative == "two-sided" else alpha))
    z_beta = z_of(power)
    delta = torch.abs(auc - 0.5)
    # crude: sample_size_group_1 ≈ ((z_alpha+z_beta) * c / delta)^2, c from variance term
    c = torch.sqrt(1.0 + r) / torch.sqrt(12.0 * r)
    sample_size_group_1 = ((z_alpha + z_beta) * c / torch.clamp(delta, min=1e-8)) ** 2
    sample_size_group_1 = torch.clamp(sample_size_group_1, min=5.0)

    sample_size_group_1_current = sample_size_group_1
    for _ in range(12):
        pwr = mann_whitney_u_test_power(
            auc,
            torch.ceil(sample_size_group_1_current),
            ratio=r,
            alpha=alpha,
            alternative=alternative,
        )
        gap = torch.clamp(power - pwr, min=-0.45, max=0.45)
        sample_size_group_1_current = torch.clamp(
            sample_size_group_1_current * (1.0 + 1.25 * gap), min=5.0, max=1e7
        )

    sample_size_group_1_output = torch.ceil(sample_size_group_1_current)
    if out is not None:
        out.copy_(sample_size_group_1_output)
        return out
    return sample_size_group_1_output
