import math

import torch
from torch import Tensor

from ._mcnemars_test_power import mcnemars_test_power


def mcnemars_test_sample_size(
    p01: Tensor,
    p10: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    two_sided: bool = True,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Required number of pairs for McNemar's test (normal approximation).

    Determines the minimum sample size needed to achieve desired statistical
    power when testing for changes in paired binary outcomes using McNemar's test.

    When to Use
    -----------
    **Traditional Statistics:**
    - Planning sample sizes for before-after intervention studies
    - Designing matched case-control studies with adequate power
    - Clinical trial planning for binary endpoint comparisons
    - Market research study design for paired preference data
    - Educational assessment planning for pre-post binary outcomes

    **Machine Learning Contexts:**
    - A/B testing design: determining required user pairs for conversion analysis
    - Model validation planning: sample sizes for paired classifier comparisons
    - Feature engineering experiments: planning data size for binary outcome changes
    - Fairness auditing: determining sample sizes for bias detection studies
    - Cross-validation design: ensuring adequate power for model comparisons
    - Active learning experiments: planning iterations for binary label improvements
    - Ensemble evaluation: determining required samples for component comparisons
    - Domain adaptation studies: planning sample sizes for cross-domain validation
    - Hyperparameter tuning: determining required samples for binary metric optimization
    - Personalization evaluation: planning sample sizes for recommendation feedback analysis

    **Use this function when:**
    - Designing experiments with paired binary outcomes
    - Both discordant probabilities (p01, p10) are estimable from pilot data
    - Primary interest is in detecting changes rather than absolute levels
    - Planning matched or within-subject designs

    **Sample size considerations:**
    - Larger differences (|p01 - p10|) require smaller sample sizes
    - Two-sided tests require larger sample sizes than one-sided
    - Higher power requirements increase needed sample size
    - Lower alpha levels increase needed sample size

    Parameters
    ----------
    p01 : Tensor
        Expected probability of discordant outcome (0→1).
    p10 : Tensor
        Expected probability of discordant outcome (1→0).
    power : float, default=0.8
        Desired statistical power.
    alpha : float, default=0.05
        Significance level.
    two_sided : bool, default=True
        Whether to use a two-sided test.

    Returns
    -------
    Tensor
        Required number of paired observations.

    Examples
    --------
    >>> p01 = torch.tensor(0.2)
    >>> p10 = torch.tensor(0.1)
    >>> mcnemars_test_sample_size(p01, p10, power=0.8)
    tensor(64.0)

    Notes
    -----
    The sample size calculation uses an iterative approach starting with
    a normal approximation:

    n ≈ (z_α + z_β)² / [4 * δ² * (p01 + p10)]

    where δ = |p01 - p10| / (p01 + p10) is the relative difference.

    The function iteratively refines this estimate using the actual
    McNemar's power calculation to ensure the desired power is achieved.
    """
    p01 = torch.atleast_1d(torch.as_tensor(p01))
    p10 = torch.atleast_1d(torch.as_tensor(p10))
    dtype = (
        torch.float64
        if (p01.dtype == torch.float64 or p10.dtype == torch.float64)
        else torch.float32
    )
    p01 = p01.to(dtype)
    p10 = p10.to(dtype)

    sqrt2 = math.sqrt(2.0)

    def z_of(prob: float) -> Tensor:
        pt = torch.tensor(prob, dtype=dtype)
        eps = torch.finfo(dtype).eps
        pt = torch.clamp(pt, min=eps, max=1 - eps)
        return sqrt2 * torch.erfinv(2.0 * pt - 1.0)

    if two_sided:
        z_alpha = z_of(1 - alpha / 2)
    else:
        z_alpha = z_of(1 - alpha)
    z_beta = z_of(power)
    probability = torch.where(
        (p01 + p10) > 0, p01 / torch.clamp(p01 + p10, min=1e-12), torch.zeros_like(p01)
    )
    delta = torch.abs(probability - 0.5)
    n0 = ((z_alpha + z_beta) / (2 * torch.clamp(delta, min=1e-8))) ** 2 / torch.clamp(
        p01 + p10, min=1e-8
    )
    n0 = torch.clamp(n0, min=4.0)

    n_curr = n0
    for _ in range(12):
        pwr = mcnemars_test_power(
            p01, p10, torch.ceil(n_curr), alpha=alpha, two_sided=two_sided
        )
        gap = torch.clamp(power - pwr, min=-0.45, max=0.45)
        n_curr = torch.clamp(n_curr * (1.0 + 1.25 * gap), min=4.0, max=1e7)

    n_out = torch.ceil(n_curr)
    if out is not None:
        out.copy_(n_out)
        return out
    return n_out
