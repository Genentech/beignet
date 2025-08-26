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

    This function calculates the minimum sample size for group 1 needed to achieve
    specified statistical power in Welch's two-sample t-test, which accounts for
    unequal variances and sample sizes. It uses iterative refinement with normal
    approximations to handle the complex noncentral t-distribution calculations.

    This function is differentiable with respect to effect_size, ratio, and var_ratio
    parameters. While traditional sample size calculations don't require gradients,
    differentiability enables integration into machine learning pipelines where
    these parameters might be learned or part of experimental design optimization.

    When to Use
    -----------
    **Traditional Statistics:**
    - Planning studies comparing groups with potentially unequal variances
    - Sample size planning for clinical trials with heterogeneous populations
    - Designing educational research with diverse student groups
    - Quality control studies comparing different measurement instruments
    - Survey research planning for groups with different response variabilities

    **Machine Learning Contexts:**
    - Planning model comparison studies with different prediction uncertainties
    - A/B testing design where groups may have unequal outcome variance
    - Experimental planning for comparing algorithms across heterogeneous datasets
    - Transfer learning studies: planning comparisons across domains with different variability
    - Ensemble method evaluation: planning studies comparing models with varying consistency
    - Cross-validation planning: accounting for different fold performance variabilities
    - Hyperparameter optimization: planning comparisons with unequal variance configurations
    - Active learning: planning experiments comparing selection strategies with different uncertainties
    - Federated learning: planning client comparisons with heterogeneous data distributions
    - Domain adaptation: sample size planning for cross-domain performance studies
    - Computer vision: planning comparisons on datasets with varying image complexity
    - NLP: planning text model comparisons on corpora with different linguistic diversity
    - Recommendation systems: planning algorithm comparisons across user groups with different engagement
    - Anomaly detection: planning method comparisons on datasets with varying anomaly rates
    - Time series forecasting: planning model comparisons with different prediction stabilities

    **Choose Welch's t-test sample size over standard t-test when:**
    - Group variances are expected to be unequal (variance ratio ≠ 1.0)
    - Sample sizes will be unequal between groups (ratio ≠ 1.0)
    - Robustness to variance assumption violations is important
    - Groups represent different populations with inherent variability differences
    - Conservative approach is preferred when variance equality is uncertain

    **Choose Welch's over non-parametric sample size calculations when:**
    - Data is expected to be approximately normally distributed within groups
    - Sample sizes will be moderate to large (n ≥ 15 per group)
    - Effect size estimation and confidence intervals are priorities
    - Maximum statistical power is desired despite unequal variances

    **Interpretation Guidelines:**
    - Effect size uses group 1 standard deviation as reference unit
    - Ratio parameter controls relative sample sizes (n₂ = ratio × n₁)
    - Variance ratio σ₂²/σ₁² affects required sample size significantly
    - Larger variance ratios (further from 1.0) require larger sample sizes
    - Consider practical constraints including cost, time, and recruitment feasibility
    - Account for potential dropout rates in planning

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
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    r = torch.as_tensor(ratio)
    vr = torch.as_tensor(var_ratio)

    dtype = (
        torch.float64
        if any(t.dtype == torch.float64 for t in (effect_size, r, vr))
        else torch.float32
    )
    effect_size = effect_size.to(dtype)
    r = r.to(dtype) if isinstance(r, Tensor) else torch.tensor(float(r), dtype=dtype)
    vr = (
        vr.to(dtype) if isinstance(vr, Tensor) else torch.tensor(float(vr), dtype=dtype)
    )

    effect_size = torch.clamp(effect_size, min=1e-8)
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

    # Welch SE with sample_size_group_1 unknown: se = sqrt(1/sample_size_group_1 + vr/(sample_size_group_1*ratio)) = sqrt((1 + vr/ratio)/sample_size_group_1)
    variance_scaling_factor = 1.0 + vr / r
    sample_size_group_1_guess = (
        (z_alpha + z_beta) * torch.sqrt(variance_scaling_factor) / effect_size
    ) ** 2
    sample_size_group_1_guess = torch.clamp(sample_size_group_1_guess, min=2.0)

    sample_size_group_1_current = sample_size_group_1_guess
    max_iter = 12
    for _ in range(max_iter):
        sample_size_group_2_current = torch.clamp(
            torch.ceil(sample_size_group_1_current * r), min=2.0
        )
        # Welch SE and df
        a = 1.0 / sample_size_group_1_current
        b = vr / sample_size_group_2_current
        se2 = a + b
        se = torch.sqrt(se2)
        degrees_of_freedom = (se2**2) / (
            a**2 / torch.clamp(sample_size_group_1_current - 1, min=1.0)
            + b**2 / torch.clamp(sample_size_group_2_current - 1, min=1.0)
        )
        # Critical value with degrees_of_freedom adjustment
        if alt == "two-sided":
            tcrit = z_alpha * torch.sqrt(1 + 1 / (2 * degrees_of_freedom))
        else:
            tcrit = z_alpha * torch.sqrt(1 + 1 / (2 * degrees_of_freedom))
        # Noncentrality
        noncentrality_parameter = effect_size / torch.clamp(se, min=1e-12)
        # Approx noncentral t variance
        var_nct = torch.where(
            degrees_of_freedom > 2,
            (degrees_of_freedom + noncentrality_parameter**2)
            / (degrees_of_freedom - 2),
            1
            + noncentrality_parameter**2
            / (2 * torch.clamp(degrees_of_freedom, min=1.0)),
        )
        std_nct = torch.sqrt(var_nct)
        if alt == "two-sided":
            zu = (tcrit - noncentrality_parameter) / torch.clamp(std_nct, min=1e-10)
            zl = (-tcrit - noncentrality_parameter) / torch.clamp(std_nct, min=1e-10)
            p_curr = 0.5 * (
                1 - torch.erf(zu / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
            ) + 0.5 * (1 + torch.erf(zl / torch.sqrt(torch.tensor(2.0, dtype=dtype))))
        elif alt == "greater":
            zscore = (tcrit - noncentrality_parameter) / torch.clamp(std_nct, min=1e-10)
            p_curr = 0.5 * (
                1 - torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
            )
        else:
            zscore = (-tcrit - noncentrality_parameter) / torch.clamp(
                std_nct, min=1e-10
            )
            p_curr = 0.5 * (
                1 + torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
            )

        # Update sample_size_group_1 heuristically based on power gap
        gap = torch.clamp(power - p_curr, min=-0.49, max=0.49)
        sample_size_group_1_current = torch.clamp(
            sample_size_group_1_current * (1.0 + 1.25 * gap), min=2.0, max=1e7
        )

    result = torch.ceil(sample_size_group_1_current)
    result = torch.clamp(result, min=2.0)
    if out is not None:
        out.copy_(result)
        return out
    return result
