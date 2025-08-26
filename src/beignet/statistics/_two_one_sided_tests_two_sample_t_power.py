import math

import torch
from torch import Tensor


def two_one_sided_tests_two_sample_t_power(
    true_effect: Tensor,
    nobs1: Tensor,
    ratio: Tensor | float | None = None,
    low: Tensor = 0.0,
    high: Tensor = 0.0,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Compute power for two-sample Two One-Sided Tests (equivalence) under equal-variance t-test.

    Parameters
    ----------
    true_effect : Tensor
        Standardized true effect d = (μ₁ − μ₂)/σ_pooled.
    nobs1 : Tensor
        Group 1 sample size.
    ratio : Tensor or float, optional
        Ratio n2/n1. If None, uses 1.0.
    low : Tensor, default=0.0
        Lower equivalence margin (standardized).
    high : Tensor, default=0.0
        Upper equivalence margin (standardized).
    alpha : float, default=0.05
        Significance level per one-sided test.

    Returns
    -------
    Tensor
        Equivalence power.

    When to Use
    -----------
    **Traditional Statistics:**

    - **Bioequivalence studies:** Testing if two treatments have equivalent effects
    - **Non-inferiority trials:** Showing new treatment is not worse than standard
    - **Method comparison studies:** Demonstrating equivalent performance of measurement methods
    - **Generic drug approval:** Establishing therapeutic equivalence to reference drugs
    - **Quality control:** Testing if process changes maintain equivalent outcomes
    - **Educational assessment:** Showing equivalent learning outcomes across different approaches

    **Machine Learning Applications:**

    - **Model equivalence testing:** Demonstrating two models perform equivalently
    - **Algorithm comparison:** Testing if new algorithm is equivalent to baseline
    - **A/B testing equivalence:** Showing treatments have equivalent effects (no difference)
    - **Feature ablation studies:** Testing if feature removal doesn't significantly harm performance
    - **Model deployment validation:** Ensuring production model performs equivalently to development
    - **Fairness testing:** Demonstrating equivalent performance across demographic groups
    - **Cross-platform validation:** Testing model equivalence across different computing environments
    - **Version control testing:** Ensuring software updates maintain equivalent performance
    - **Distributed system validation:** Testing equivalent performance across different nodes
    - **Resource optimization:** Showing reduced resources maintain equivalent performance
    - **Compression evaluation:** Testing if model compression preserves equivalent performance
    - **Transfer learning validation:** Demonstrating equivalent performance in new domains
    - **Ensemble component testing:** Testing if ensemble components contribute equivalently
    - **Hyperparameter robustness:** Showing equivalent performance across parameter ranges
    - **Data preprocessing comparison:** Testing equivalent outcomes from different preprocessing

    **Interpretation Guidelines:**

    - **Equivalence margins:** Define practically meaningful difference thresholds
    - **Power interpretation:** High power means strong evidence of equivalence if effect is small
    - **Two one-sided tests:** Both tests must be significant to conclude equivalence
    - **Effect size consideration:** True effect should be close to zero for meaningful equivalence
    - **Margin selection:** Equivalence margins should be based on practical significance, not statistical convenience
    - **Sample size requirements:** Equivalence testing typically requires larger samples than superiority testing
    """
    d = torch.atleast_1d(torch.as_tensor(true_effect))
    n1 = torch.atleast_1d(torch.as_tensor(nobs1))
    if ratio is None:
        ratio_t = torch.tensor(1.0)
    else:
        ratio_t = torch.atleast_1d(torch.as_tensor(ratio))
    low = torch.atleast_1d(torch.as_tensor(low))
    high = torch.atleast_1d(torch.as_tensor(high))

    dtype = (
        torch.float64
        if any(t.dtype == torch.float64 for t in (d, n1, ratio_t, low, high))
        else torch.float32
    )
    d = d.to(dtype)
    n1 = n1.to(dtype)
    ratio_t = ratio_t.to(dtype)
    low = low.to(dtype)
    high = high.to(dtype)

    n1 = torch.clamp(n1, min=2.0)
    ratio_t = torch.clamp(ratio_t, min=0.1, max=10.0)
    n2 = n1 * ratio_t
    df = torch.clamp(n1 + n2 - 2, min=1.0)

    se_factor = torch.sqrt(1.0 / n1 + 1.0 / n2)
    ncp_low = (d - low) / torch.clamp(se_factor, min=1e-12)
    ncp_high = (d - high) / torch.clamp(se_factor, min=1e-12)

    sqrt2 = math.sqrt(2.0)
    z = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
    tcrit = z * torch.sqrt(1 + 1 / (2 * df))

    def power_greater(ncp: Tensor) -> Tensor:
        var = torch.where(
            df > 2,
            (df + ncp**2) / (df - 2),
            1 + ncp**2 / (2 * torch.clamp(df, min=1.0)),
        )
        std = torch.sqrt(var)
        zscore = (tcrit - ncp) / torch.clamp(std, min=1e-10)
        return 0.5 * (
            1 - torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
        )

    def power_less(ncp: Tensor) -> Tensor:
        var = torch.where(
            df > 2,
            (df + ncp**2) / (df - 2),
            1 + ncp**2 / (2 * torch.clamp(df, min=1.0)),
        )
        std = torch.sqrt(var)
        zscore = (-tcrit - ncp) / torch.clamp(std, min=1e-10)
        return 0.5 * (
            1 + torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
        )

    p_lower = power_greater(ncp_low)
    p_upper = power_less(ncp_high)
    power = torch.minimum(p_lower, p_upper)
    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)
        return out
    return power
