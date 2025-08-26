import math

import torch
from torch import Tensor


def paired_t_test_power(
    effect_size: Tensor,
    sample_size: Tensor,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Power for paired-samples t-test (unknown variance of differences).

    This function calculates the probability of correctly rejecting the null
    hypothesis when the alternative hypothesis is true for a paired-samples t-test.
    It accounts for the correlation between paired observations, making it more
    powerful than independent samples tests when the pairing is effective.

    This function is differentiable with respect to all tensor parameters.
    While traditional power analysis doesn't require gradients, differentiability
    enables integration into machine learning pipelines where effect sizes or
    sample sizes might be learned parameters or part of experimental design
    optimization.

    When to Use
    -----------
    **Traditional Statistics:**
    - Before-after treatment comparisons in clinical trials
    - Pre-test/post-test designs in educational research
    - Matched pairs experimental designs
    - Crossover studies with repeated measures
    - Longitudinal studies measuring change over time

    **Machine Learning Contexts:**
    - Comparing model performance on the same validation sets
    - A/B testing with matched user pairs or repeated measures
    - Evaluating training improvements across ML model iterations
    - Cross-validation: comparing algorithms on identical data splits
    - Transfer learning: measuring performance gains when fine-tuning models
    - Hyperparameter optimization: paired comparisons of configurations
    - Ensemble methods: comparing individual vs. ensemble performance
    - Active learning: measuring information gain from new training samples
    - Federated learning: comparing local vs. global model performance
    - Time series analysis: before/after intervention effects in forecasting
    - Computer vision: comparing model performance on same image sets
    - NLP: evaluating model improvements on identical text corpora
    - Recommendation systems: A/B testing with user-matched designs
    - Anomaly detection: comparing detection rates before/after model updates
    - Causal inference: measuring treatment effects in matched observational data

    **Choose paired t-test power over independent t-test power when:**
    - Same subjects/units measured twice (e.g., model performance on same data)
    - Matched pairs design reduces between-subject variability
    - Correlation between measurements is moderate to high (r > 0.3)
    - Want to detect smaller effect sizes with improved statistical power
    - Controlling for confounding variables through matching

    **Choose paired t-test over non-parametric alternatives when:**
    - Differences are approximately normally distributed
    - Sample size is moderate to large (n ≥ 15 pairs)
    - Maximum statistical power is desired
    - Effect size estimation and confidence intervals are needed

    **Interpretation Guidelines:**
    - Higher correlation between pairs increases power dramatically
    - Effect size refers to standardized mean difference of paired differences
    - Power increases with larger effect sizes and sample sizes
    - Consider practical significance alongside statistical power

    Parameters
    ----------
    effect_size : Tensor
        Standardized mean difference of pairs d = μ_d/σ_d.
    sample_size : Tensor
        Number of pairs (n >= 2).
    alpha : float, default=0.05
        Significance level.
    alternative : {"two-sided", "greater", "less"}
        Alternative hypothesis.

    Returns
    -------
    Tensor
        Statistical power.
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))
    dtype = (
        torch.float64
        if (effect_size.dtype == torch.float64 or sample_size.dtype == torch.float64)
        else torch.float32
    )
    effect_size = effect_size.to(dtype)
    sample_size = torch.clamp(sample_size.to(dtype), min=2.0)

    degrees_of_freedom = sample_size - 1
    noncentrality_parameter = effect_size * torch.sqrt(sample_size)

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    sqrt2 = math.sqrt(2.0)
    if alt == "two-sided":
        z = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt2
        tcrit = z * torch.sqrt(1 + 1 / (2 * degrees_of_freedom))
    else:
        z = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
        tcrit = z * torch.sqrt(1 + 1 / (2 * degrees_of_freedom))

    var_nct = torch.where(
        degrees_of_freedom > 2,
        (degrees_of_freedom + noncentrality_parameter**2) / (degrees_of_freedom - 2),
        1 + noncentrality_parameter**2 / (2 * torch.clamp(degrees_of_freedom, min=1.0)),
    )
    std_nct = torch.sqrt(var_nct)

    if alt == "two-sided":
        zu = (tcrit - noncentrality_parameter) / torch.clamp(std_nct, min=1e-10)
        zl = (-tcrit - noncentrality_parameter) / torch.clamp(std_nct, min=1e-10)
        power = 0.5 * (
            1 - torch.erf(zu / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
        ) + 0.5 * (1 + torch.erf(zl / torch.sqrt(torch.tensor(2.0, dtype=dtype))))
    elif alt == "greater":
        zscore = (tcrit - noncentrality_parameter) / torch.clamp(std_nct, min=1e-10)
        power = 0.5 * (
            1 - torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
        )
    else:
        zscore = (-tcrit - noncentrality_parameter) / torch.clamp(std_nct, min=1e-10)
        power = 0.5 * (
            1 + torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
        )

    out_t = torch.clamp(power, 0.0, 1.0)
    if out is not None:
        out.copy_(out_t)
        return out
    return out_t
