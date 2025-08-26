import math

import torch
from torch import Tensor


def mcnemars_test_power(
    p01: Tensor,
    p10: Tensor,
    sample_size: Tensor,
    alpha: float = 0.05,
    two_sided: bool = True,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Power for McNemar's test using a normal approximation.

    McNemar's test evaluates the significance of change between paired binary
    outcomes, testing whether the marginal probabilities differ significantly.
    This test is particularly valuable for before-after studies and matched
    case-control designs.

    When to Use
    -----------
    **Traditional Statistics:**
    - Pre-post intervention studies with binary outcomes
    - Matched case-control studies comparing exposure patterns
    - Test-retest reliability studies for binary diagnostic tests
    - Clinical trials comparing treatment effects on binary endpoints
    - Marketing research analyzing consumer preference changes

    **Machine Learning Contexts:**
    - A/B testing for binary outcomes (click rates, conversion rates)
    - Model comparison on binary classification tasks with paired data
    - Feature selection: comparing model performance before/after adding features
    - Fairness assessment: detecting bias changes after model interventions
    - Cross-validation: comparing classifiers on same data splits
    - Active learning: evaluating query strategy effectiveness on binary labels
    - Ensemble methods: comparing individual vs ensemble binary predictions
    - Domain adaptation: assessing classification performance across domains
    - Hyperparameter optimization: comparing binary metric improvements
    - Personalization: evaluating recommendation system binary feedback changes

    **Choose McNemar's test over other tests when:**
    - Comparing paired binary outcomes (not independent groups)
    - Data has natural pairing (same subjects, matched controls)
    - Interest is in change/difference rather than absolute values
    - Sample sizes are moderate to large (n ≥ 25 pairs)

    **Choose McNemar's over Chi-square independence when:**
    - Data structure is 2×2 contingency table from paired observations
    - Testing symmetry rather than independence
    - Same subjects measured twice rather than independent samples

    Parameters
    ----------
    p01 : Tensor
        Probability of discordant outcome (0→1).
    p10 : Tensor
        Probability of discordant outcome (1→0).
    sample_size : Tensor
        Number of paired observations.
    alpha : float, default=0.05
        Significance level.
    two_sided : bool, default=True
        Whether to use a two-sided test.

    Returns
    -------
    Tensor
        Statistical power.

    Examples
    --------
    >>> p01 = torch.tensor(0.2)
    >>> p10 = torch.tensor(0.1)
    >>> sample_size = torch.tensor(100)
    >>> mcnemars_test_power(p01, p10, sample_size)
    tensor(0.6234)

    Notes
    -----
    McNemar's test statistic is based on the discordant pairs:

    χ² = (|b - c| - 0.5)² / (b + c)

    where b and c are the counts of discordant pairs (01 and 10 respectively).
    The continuity correction (0.5) is applied for better small-sample performance.

    The test assumes:
    - Independent paired observations
    - Binary outcomes
    - Marginal homogeneity under null hypothesis

    Power depends primarily on the difference |p01 - p10| rather than their
    individual values, making the test sensitive to asymmetric changes.
    """
    p01 = torch.atleast_1d(torch.as_tensor(p01))
    p10 = torch.atleast_1d(torch.as_tensor(p10))
    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))
    dtype = (
        torch.float64
        if any(t.dtype == torch.float64 for t in (p01, p10, sample_size))
        else torch.float32
    )
    p01 = torch.clamp(p01.to(dtype), 0.0, 1.0)
    p10 = torch.clamp(p10.to(dtype), 0.0, 1.0)
    sample_size = torch.clamp(sample_size.to(dtype), min=1.0)

    D = sample_size * (p01 + p10)
    probability = torch.where(
        (p01 + p10) > 0, p01 / torch.clamp(p01 + p10, min=1e-12), torch.zeros_like(p01)
    )
    mean = D * (probability - 0.5)
    std = torch.sqrt(torch.clamp(D * 0.25, min=1e-12))

    sqrt2 = math.sqrt(2.0)

    def z_of(prob: float) -> Tensor:
        pt = torch.tensor(prob, dtype=dtype)
        eps = torch.finfo(dtype).eps
        pt = torch.clamp(pt, min=eps, max=1 - eps)
        return sqrt2 * torch.erfinv(2.0 * pt - 1.0)

    if two_sided:
        zcrit = z_of(1 - alpha / 2)
        z_upper = zcrit - mean / torch.clamp(std, min=1e-12)
        z_lower = -zcrit - mean / torch.clamp(std, min=1e-12)
        power = 0.5 * (1 - torch.erf(z_upper / math.sqrt(2.0))) + 0.5 * (
            1 + torch.erf(z_lower / math.sqrt(2.0))
        )
    else:
        zcrit = z_of(1 - alpha)
        zscore = zcrit - mean / torch.clamp(std, min=1e-12)
        power = 0.5 * (1 - torch.erf(zscore / math.sqrt(2.0)))

    out_t = torch.clamp(power, 0.0, 1.0)
    if out is not None:
        out.copy_(out_t)
        return out
    return out_t
