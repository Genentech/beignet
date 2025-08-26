import math

import torch
from torch import Tensor

from ._mann_whitney_u_test_power import mann_whitney_u_test_power


def mann_whitney_u_test_minimum_detectable_effect(
    nobs1: Tensor,
    nobs2: Tensor | None = None,
    ratio: Tensor | float = 1.0,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Minimum detectable AUC (P(X>Y)+0.5 P(X=Y)) for Mann–Whitney U test.

    Parameters
    ----------
    nobs1 : Tensor
        Sample size of group 1.
    nobs2 : Tensor, optional
        Sample size of group 2. If None, uses ceil(ratio * nobs1).
    ratio : Tensor or float, default=1.0
        nobs2/nobs1 when nobs2 is None.
    power : float, default=0.8
        Target power.
    alpha : float, default=0.05
        Significance level.
    alternative : {"two-sided", "greater", "less"}
        Test direction; returns AUC ≥ 0.5 for "greater"/"two-sided" and ≤ 0.5 for "less".

    Returns
    -------
    Tensor
        Minimal detectable AUC value.

    When to Use
    -----------
    **Traditional Statistics:**

    - **Non-parametric comparisons:** When distributions are non-normal or ordinal data
    - **Robust effect size planning:** Alternative to t-test when outliers are expected
    - **Rank-based analysis:** When interested in relative ordering rather than exact values
    - **Small sample studies:** When sample sizes are too small for normality assumptions
    - **Skewed data analysis:** When data distributions are heavily skewed or multi-modal
    - **Ordinal scale research:** Surveys, ratings, or ranked preference data

    **Machine Learning Applications:**

    - **Binary classification evaluation:** Minimum detectable AUC improvements in ROC analysis
    - **Ranking algorithm assessment:** Minimum detectable differences in ranking quality
    - **Recommendation system testing:** AUC thresholds for recommendation performance
    - **Anomaly detection validation:** Minimum detectable discrimination between normal/anomalous
    - **Model comparison (non-parametric):** Comparing models without normality assumptions
    - **Feature importance ranking:** Minimum detectable differences in feature discrimination
    - **A/B testing with ordinal outcomes:** Testing rank-based metrics like user satisfaction scores
    - **Information retrieval evaluation:** Minimum detectable improvements in search result quality
    - **Fairness testing:** Detecting discriminatory ranking patterns across demographic groups
    - **Time series anomaly detection:** Minimum detectable shifts in temporal ranking patterns
    - **Multi-class classification:** Pairwise AUC analysis between class predictions
    - **Ensemble method evaluation:** Rank correlation analysis between ensemble components
    - **Active learning strategies:** Minimum detectable improvements in sample selection quality
    - **Transfer learning assessment:** Rank preservation across different domains
    - **Clustering validation:** Minimum detectable differences in cluster separation quality

    **Interpretation Guidelines:**

    - **AUC = 0.5:** No discrimination ability (random performance)
    - **AUC = 0.6:** Small effect (weak discrimination)
    - **AUC = 0.7:** Medium effect (moderate discrimination)
    - **AUC = 0.8:** Large effect (strong discrimination)
    - **AUC > 0.9:** Very large effect (excellent discrimination)
    - **Consider practical context:** Domain-specific thresholds may differ from statistical conventions
    """
    sample_size_group_1_0 = torch.as_tensor(nobs1)
    scalar_out = sample_size_group_1_0.ndim == 0
    sample_size_group_1 = torch.atleast_1d(sample_size_group_1_0)
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
        sample_size_group_2_0 = torch.as_tensor(nobs2)
        scalar_out = scalar_out and (sample_size_group_2_0.ndim == 0)
        sample_size_group_2 = torch.atleast_1d(sample_size_group_2_0)

    dtype = (
        torch.float64
        if any(
            t.dtype == torch.float64 for t in (sample_size_group_1, sample_size_group_2)
        )
        else torch.float32
    )
    sample_size_group_1 = torch.clamp(sample_size_group_1.to(dtype), min=2.0)
    sample_size_group_2 = torch.clamp(sample_size_group_2.to(dtype), min=2.0)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    sqrt2 = math.sqrt(2.0)

    def z_of(p: float) -> Tensor:
        q = torch.tensor(p, dtype=dtype)
        eps = torch.finfo(dtype).eps
        q = torch.clamp(q, min=eps, max=1 - eps)
        return sqrt2 * torch.erfinv(2.0 * q - 1.0)

    z_alpha = z_of(1 - alpha / 2) if alt == "two-sided" else z_of(1 - alpha)
    z_beta = z_of(power)
    scale = torch.sqrt(
        12.0
        * sample_size_group_1
        * sample_size_group_2
        / (sample_size_group_1 + sample_size_group_2 + 1.0)
    )
    delta = (z_alpha + z_beta) / torch.clamp(scale, min=1e-12)

    if alt == "less":
        auc = torch.clamp(0.5 - delta, min=0.0)
    else:
        auc = torch.clamp(0.5 + delta, max=1.0)

    p_curr = mann_whitney_u_test_power(
        auc, sample_size_group_1, sample_size_group_2, alpha=alpha, alternative=alt
    )
    gap = torch.clamp(power - p_curr, min=-0.45, max=0.45)
    step = gap * 0.05
    if alt == "less":
        auc = torch.clamp(auc - torch.abs(step), 0.0, 1.0)
    else:
        auc = torch.clamp(auc + torch.abs(step), 0.0, 1.0)

    if scalar_out:
        auc_s = auc.reshape(())
        if out is not None:
            out.copy_(auc_s)
            return out
        return auc_s
    else:
        if out is not None:
            out.copy_(auc)
            return out
        return auc
