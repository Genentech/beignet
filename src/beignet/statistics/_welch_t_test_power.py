import math

import torch
from torch import Tensor


def welch_t_test_power(
    effect_size: Tensor,
    nobs1: Tensor,
    nobs2: Tensor,
    var_ratio: Tensor | float = 1.0,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute statistical power for Welch's two-sample t-test.

    Welch's t-test allows unequal variances and sample sizes, making it more
    robust than the standard independent samples t-test when the assumption
    of equal variances is violated. This function calculates the probability
    of correctly rejecting the null hypothesis when the alternative is true.

    This function is differentiable with respect to all tensor parameters.
    While traditional power analysis doesn't require gradients, differentiability
    enables integration into machine learning pipelines where effect sizes or
    sample sizes might be learned parameters or part of experimental design
    optimization.

    When to Use
    -----------
    **Traditional Statistics:**
    - Comparing groups with potentially unequal variances
    - Clinical trials where treatment and control groups have different variability
    - Educational research comparing diverse student populations
    - Quality control comparing measurements from different instruments
    - Survey research with groups of different sizes and variability

    **Machine Learning Contexts:**
    - Comparing ML models with different prediction uncertainty patterns
    - A/B testing where groups have unequal variance in outcomes
    - Model performance comparison across heterogeneous datasets
    - Transfer learning: comparing performance with different domain variability
    - Ensemble methods: comparing individual models with varying consistency
    - Cross-validation: accounting for different fold performance variability
    - Hyperparameter optimization: comparing configurations with unequal variance
    - Active learning: comparing selection strategies with different uncertainty
    - Federated learning: comparing clients with heterogeneous data distributions
    - Domain adaptation: power analysis for models across different data sources
    - Computer vision: comparing models on datasets with varying image complexity
    - NLP: comparing text models on corpora with different linguistic diversity
    - Recommendation systems: comparing algorithms across user groups with different engagement patterns
    - Anomaly detection: comparing methods on datasets with varying anomaly rates
    - Time series forecasting: comparing models with different prediction stability

    **Choose Welch's t-test power over standard t-test power when:**
    - Group variances are expected to be unequal (variance ratio > 2 or < 0.5)
    - Sample sizes are unequal between groups
    - Robustness to variance assumption violations is important
    - Groups represent different populations with inherent variability differences
    - Uncertainty about meeting equal variance assumptions

    **Choose Welch's t-test over non-parametric alternatives when:**
    - Data is approximately normally distributed within groups
    - Sample sizes are moderate to large (n ≥ 15 per group)
    - Effect size estimation and confidence intervals are needed
    - Maximum statistical power is desired despite unequal variances

    **Interpretation Guidelines:**
    - Effect size uses group 1 standard deviation as reference
    - Variance ratio σ₂²/σ₁² = 1.0 reduces to standard t-test
    - Welch's test adjusts degrees of freedom for unequal variances
    - Power decreases with larger variance ratios (further from 1.0)
    - Consider practical significance alongside statistical power

    Parameters
    ----------
    effect_size : Tensor
        Standardized effect size d = (μ₁ − μ₂) / σ₁ using group 1 SD as unit.
        Should be non-negative for power (direction handled by `alternative`).

    nobs1 : Tensor
        Sample size for group 1.

    nobs2 : Tensor
        Sample size for group 2.

    var_ratio : Tensor or float, default=1.0
        Variance ratio σ₂²/σ₁². Use 1.0 for equal variances.

    alpha : float, default=0.05
        Significance level (Type I error rate).

    alternative : str, default="two-sided"
        Type of alternative hypothesis: "two-sided", "greater", or "less".

    out : Tensor, optional
        Output tensor. Default, `None`.

    Returns
    -------
    output : Tensor
        Statistical power (probability of correctly rejecting false null hypothesis).
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    sample_size_group_1 = torch.atleast_1d(torch.as_tensor(nobs1))
    sample_size_group_2 = torch.atleast_1d(torch.as_tensor(nobs2))
    vr = torch.as_tensor(var_ratio)

    # Dtype
    if any(
        t.dtype == torch.float64
        for t in (
            effect_size,
            sample_size_group_1,
            sample_size_group_2,
            vr if isinstance(vr, Tensor) else torch.tensor(0.0),
        )
    ):
        dtype = torch.float64
    else:
        dtype = torch.float32
    effect_size = effect_size.to(dtype)
    sample_size_group_1 = sample_size_group_1.to(dtype)
    sample_size_group_2 = sample_size_group_2.to(dtype)
    if isinstance(vr, Tensor):
        vr = vr.to(dtype)
    else:
        vr = torch.tensor(float(vr), dtype=dtype)

    # Clamp
    effect_size = torch.clamp(effect_size, min=0.0)
    sample_size_group_1 = torch.clamp(sample_size_group_1, min=2.0)
    sample_size_group_2 = torch.clamp(sample_size_group_2, min=2.0)
    vr = torch.clamp(vr, min=1e-6, max=1e6)

    # Welch SE and df
    a = 1.0 / sample_size_group_1
    b = vr / sample_size_group_2
    se2 = a + b
    se = torch.sqrt(se2)
    degrees_of_freedom = (se2**2) / (
        a**2 / torch.clamp(sample_size_group_1 - 1, min=1.0)
        + b**2 / torch.clamp(sample_size_group_2 - 1, min=1.0)
    )

    # Noncentrality parameter
    ncp = effect_size / torch.clamp(se, min=1e-12)

    # Alternative normalization
    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    # Critical value approx (normal with degrees_of_freedom adjustment)
    sqrt2 = math.sqrt(2.0)
    if alt == "two-sided":
        z = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt2
        tcrit = z * torch.sqrt(1 + 1 / (2 * degrees_of_freedom))
    else:
        z = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
        tcrit = z * torch.sqrt(1 + 1 / (2 * degrees_of_freedom))

    # Approximate noncentral t by normal with mean=ncp and var=(degrees_of_freedom+ncp^2)/(degrees_of_freedom-2)
    var_nct = torch.where(
        degrees_of_freedom > 2,
        (degrees_of_freedom + ncp**2) / (degrees_of_freedom - 2),
        1 + ncp**2 / (2 * torch.clamp(degrees_of_freedom, min=1.0)),
    )
    std_nct = torch.sqrt(var_nct)

    if alt == "two-sided":
        zu = (tcrit - ncp) / torch.clamp(std_nct, min=1e-10)
        zl = (-tcrit - ncp) / torch.clamp(std_nct, min=1e-10)
        power = 0.5 * (
            1 - torch.erf(zu / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
        ) + 0.5 * (1 + torch.erf(zl / torch.sqrt(torch.tensor(2.0, dtype=dtype))))
    elif alt == "greater":
        zscore = (tcrit - ncp) / torch.clamp(std_nct, min=1e-10)
        power = 0.5 * (
            1 - torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
        )
    else:  # less
        zscore = (-tcrit - ncp) / torch.clamp(std_nct, min=1e-10)
        power = 0.5 * (
            1 + torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
        )

    output = torch.clamp(power, 0.0, 1.0)
    if out is not None:
        out.copy_(output)
        return out
    return output
