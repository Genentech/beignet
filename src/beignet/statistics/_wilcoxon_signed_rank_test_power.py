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

    When to Use
    -----------
    **Traditional Statistics:**
    - Paired data comparisons when normality assumptions are violated
    - Pre-post treatment comparisons with non-normal outcomes
    - Matched pairs design with ordinal or skewed continuous data
    - Small sample sizes where parametric assumptions fail

    **Machine Learning Contexts:**
    - Paired model comparison: before/after model updates on same data
    - Cross-validation: comparing model performance across paired folds
    - A/B testing with paired users (before/after treatment)
    - Feature engineering: comparing model performance with/without features on same data
    - Hyperparameter tuning: paired comparisons of parameter settings
    - Transfer learning: comparing pre-trained vs. fine-tuned models
    - Time series: comparing model performance across paired time periods
    - Recommendation systems: paired user preference comparisons
    - Medical AI: paired diagnostic comparisons (human vs. AI on same cases)

    **Choose Wilcoxon signed-rank over paired t-test when:**
    - Non-normal paired differences
    - Ordinal outcome data
    - Presence of outliers in differences
    - Small sample sizes (n < 30)
    - Robust analysis desired

    **Probability Interpretation:**
    - prob_positive = 0.5: no difference (null hypothesis)
    - prob_positive > 0.5: positive treatment effect
    - prob_positive < 0.5: negative treatment effect
    - prob_positive = 0.6: small effect
    - prob_positive = 0.7: medium effect
    - prob_positive = 0.8: large effect

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
    probability = torch.atleast_1d(torch.as_tensor(prob_positive))
    sample_size = torch.atleast_1d(torch.as_tensor(nobs))

    dtype = (
        torch.float64
        if (probability.dtype == torch.float64 or sample_size.dtype == torch.float64)
        else torch.float32
    )
    probability = probability.to(dtype)
    sample_size = torch.clamp(sample_size.to(dtype), min=5.0)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    # Sum of ranks S and H0 moments for W+
    S = sample_size * (sample_size + 1.0) / 2.0
    mean0 = S / 2.0
    var0 = sample_size * (sample_size + 1.0) * (2.0 * sample_size + 1.0) / 24.0
    sd0 = torch.sqrt(torch.clamp(var0, min=1e-12))

    # H1 mean using prob_positive; variance approx by var0
    mean1 = S * probability
    noncentrality_parameter = (mean1 - mean0) / sd0

    sqrt2 = math.sqrt(2.0)

    def z_of(prob: float) -> Tensor:
        q = torch.tensor(prob, dtype=dtype)
        eps = torch.finfo(dtype).eps
        q = torch.clamp(q, min=eps, max=1 - eps)
        return sqrt2 * torch.erfinv(2.0 * q - 1.0)

    if alt == "two-sided":
        zcrit = z_of(1 - alpha / 2)
        upper = 0.5 * (1 - torch.erf((zcrit - noncentrality_parameter) / sqrt2))
        lower = 0.5 * (1 + torch.erf((-zcrit - noncentrality_parameter) / sqrt2))
        power = upper + lower
    elif alt == "greater":
        zcrit = z_of(1 - alpha)
        power = 0.5 * (1 - torch.erf((zcrit - noncentrality_parameter) / sqrt2))
    else:
        zcrit = z_of(1 - alpha)
        power = 0.5 * (1 + torch.erf((-zcrit - noncentrality_parameter) / sqrt2))

    out_t = torch.clamp(power, 0.0, 1.0)
    if out is not None:
        out.copy_(out_t)
        return out
    return out_t
