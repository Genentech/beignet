import math

import torch
from torch import Tensor


def paired_t_test_sample_size(
    effect_size: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Required sample size (number of pairs) for paired t-test.

    This function calculates the minimum number of paired observations needed
    to achieve a specified statistical power for detecting a given effect size
    in a paired-samples t-test. It accounts for the increased efficiency that
    comes from pairing correlated observations.

    This function is differentiable with respect to the effect_size parameter.
    While traditional sample size calculations don't require gradients,
    differentiability enables integration into machine learning pipelines where
    effect sizes might be learned parameters or part of experimental design
    optimization.

    When to Use
    -----------
    **Traditional Statistics:**
    - Planning before-after treatment studies in clinical research
    - Designing pre-test/post-test educational interventions
    - Sample size planning for matched pairs experiments
    - Crossover trial design with repeated measures
    - Longitudinal study planning for measuring change over time

    **Machine Learning Contexts:**
    - Planning validation studies for ML model comparisons
    - A/B test design with matched user pairs or repeated measures
    - Experimental design for evaluating model training improvements
    - Cross-validation planning: sample sizes for algorithm comparisons
    - Transfer learning studies: measuring performance gains from fine-tuning
    - Hyperparameter optimization: planning paired comparison experiments
    - Ensemble method evaluation: comparing individual vs. ensemble performance
    - Active learning: planning experiments for measuring information gain
    - Federated learning: designing local vs. global performance comparisons
    - Time series forecasting: planning intervention effect studies
    - Computer vision: planning model performance studies on identical image sets
    - NLP: designing model improvement studies on identical text corpora
    - Recommendation systems: planning A/B tests with user-matched designs
    - Anomaly detection: planning studies for model update effectiveness
    - Causal inference: sample size planning for matched observational studies

    **Choose paired t-test sample size over independent t-test when:**
    - Same subjects/units will be measured twice (e.g., same data for model comparison)
    - Matched pairs design can reduce between-subject variability
    - Expected correlation between measurements is moderate to high (r > 0.3)
    - Want to minimize required sample size while maintaining power
    - Controlling for confounding variables through matching

    **Choose paired over non-parametric sample size calculations when:**
    - Paired differences will be approximately normally distributed
    - Maximum statistical power is desired for fixed sample size
    - Confidence intervals and effect size estimation are important
    - Parametric assumptions are likely to be met

    **Interpretation Guidelines:**
    - Effect size refers to standardized mean difference of paired differences
    - Higher expected correlation between pairs reduces required sample size
    - Consider practical constraints (e.g., dropout rates, measurement costs)
    - Account for multiple comparisons if testing multiple outcomes
    - Sample size increases dramatically for detecting small effect sizes

    Parameters
    ----------
    effect_size : Tensor
        Standardized mean difference of pairs d = μ_d/σ_d. (d > 0).
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

    if alt == "two-sided":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt2
    else:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2

    sample_size = ((z_alpha + z_beta) / effect_size) ** 2
    sample_size = torch.clamp(sample_size, min=2.0)

    sample_size_curr = sample_size
    for _ in range(10):
        degrees_of_freedom = torch.clamp(sample_size_curr - 1, min=1.0)
        tcrit = z_alpha * torch.sqrt(1 + 1 / (2 * degrees_of_freedom))
        noncentrality_parameter = effect_size * torch.sqrt(sample_size_curr)
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
            current_power = 0.5 * (
                1 - torch.erf(zu / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
            ) + 0.5 * (1 + torch.erf(zl / torch.sqrt(torch.tensor(2.0, dtype=dtype))))
        elif alt == "greater":
            zscore = (tcrit - noncentrality_parameter) / torch.clamp(std_nct, min=1e-10)
            current_power = 0.5 * (
                1 - torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
            )
        else:
            zscore = (-tcrit - noncentrality_parameter) / torch.clamp(
                std_nct, min=1e-10
            )
            current_power = 0.5 * (
                1 + torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
            )
        gap = torch.clamp(power - current_power, min=-0.45, max=0.45)
        sample_size_curr = torch.clamp(
            sample_size_curr * (1.0 + 1.25 * gap), min=2.0, max=1e7
        )

    sample_size_out = torch.ceil(sample_size_curr)
    if out is not None:
        out.copy_(sample_size_out)
        return out
    return sample_size_out
