import math

import torch
from torch import Tensor


def two_one_sided_tests_one_sample_t_power(
    true_effect: Tensor,
    sample_size: Tensor,
    low: Tensor,
    high: Tensor,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Compute power for one-sample Two One-Sided Tests (equivalence) with standardized margins.

    The Two One-Sided Tests (TOST) procedure tests for equivalence rather than
    difference, establishing that a treatment effect falls within a specified
    margin of practical equivalence. This is particularly important when the
    goal is to show that treatments are similar rather than different.

    This function is differentiable with respect to all tensor parameters.
    While traditional power analysis doesn't require gradients, differentiability
    enables integration into machine learning pipelines where effect sizes or
    equivalence margins might be learned parameters.

    When to Use
    -----------
    **Traditional Statistics:**
    - Bioequivalence studies: proving generic drugs are equivalent to branded drugs
    - Clinical trials: showing new treatment is equivalent to standard treatment
    - Quality control: proving process changes don't meaningfully affect outcomes
    - Educational research: showing alternative teaching methods are equivalent
    - Psychology: demonstrating intervention effects are within negligible range
    - Regulatory studies: proving safety/efficacy equivalence for approval

    **Machine Learning Contexts:**
    - Model comparison: proving new model performs equivalently to baseline
    - A/B testing: showing new algorithm doesn't hurt existing performance
    - Fairness evaluation: proving model performs equivalently across groups
    - Transfer learning: showing performance is maintained across domains
    - Model compression: proving compressed model is equivalent to full model
    - Active learning: showing reduced samples maintain equivalent performance
    - Cross-validation: proving performance is stable across different partitions
    - Hyperparameter optimization: showing robustness within parameter ranges
    - Ensemble methods: proving individual components contribute equivalently
    - Computer vision: showing model equivalence across different image conditions
    - NLP: showing language model equivalence across different text domains
    - Recommendation systems: proving algorithm changes don't hurt user experience

    **Choose TOST over traditional t-tests when:**
    - Research goal is to prove equivalence rather than detect difference
    - Want to establish that effect is practically negligible
    - Need to show non-inferiority or equivalence for regulatory approval
    - Testing whether change/intervention has no meaningful impact
    - Want to shift burden of proof from rejecting H₀ to proving equivalence

    **Choose one-sample TOST over two-sample when:**
    - Comparing single treatment group to known standard/reference value
    - Historical control or established benchmark is available
    - Interested in absolute equivalence to specific target value
    - Single group design is more feasible than controlled comparison

    **Interpretation Guidelines:**
    - Power represents probability of proving equivalence when it truly exists
    - Requires specifying equivalence margins (low, high) a priori
    - Both one-sided tests must be significant to conclude equivalence
    - Narrower equivalence margins require larger sample sizes
    - True effect closer to margin boundaries reduces power
    - Effect exactly at boundaries gives power ≈ α (low power region)

    Parameters
    ----------
    true_effect : Tensor
        Standardized true effect d = (μ - μ0)/σ.
    sample_size : Tensor
        Sample size n.
    low : Tensor
        Lower equivalence margin (standardized).
    high : Tensor
        Upper equivalence margin (standardized).
    alpha : float, default=0.05
        Significance level for each one-sided test.

    Returns
    -------
    Tensor
        Equivalence power (probability both one-sided tests reject).
    """
    true_effect_size = torch.atleast_1d(torch.as_tensor(true_effect))
    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))
    low = torch.atleast_1d(torch.as_tensor(low))
    high = torch.atleast_1d(torch.as_tensor(high))

    dtype = (
        torch.float64
        if any(
            t.dtype == torch.float64 for t in (true_effect_size, sample_size, low, high)
        )
        else torch.float32
    )
    true_effect_size = true_effect_size.to(dtype)
    sample_size = sample_size.to(dtype)
    low = low.to(dtype)
    high = high.to(dtype)

    sample_size = torch.clamp(sample_size, min=2.0)
    degrees_of_freedom = sample_size - 1
    ncp_low = (true_effect_size - low) * torch.sqrt(sample_size)
    ncp_high = (true_effect_size - high) * torch.sqrt(sample_size)

    sqrt2 = math.sqrt(2.0)
    z = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
    tcrit = z * torch.sqrt(1 + 1 / (2 * degrees_of_freedom))

    def power_greater(ncp: Tensor) -> Tensor:
        var = torch.where(
            degrees_of_freedom > 2,
            (degrees_of_freedom + ncp**2) / (degrees_of_freedom - 2),
            1 + ncp**2 / (2 * torch.clamp(degrees_of_freedom, min=1.0)),
        )
        std = torch.sqrt(var)
        zscore = (tcrit - ncp) / torch.clamp(std, min=1e-10)
        return 0.5 * (
            1 - torch.erf(zscore / torch.sqrt(torch.tensor(2.0, dtype=dtype)))
        )

    def power_less(ncp: Tensor) -> Tensor:
        var = torch.where(
            degrees_of_freedom > 2,
            (degrees_of_freedom + ncp**2) / (degrees_of_freedom - 2),
            1 + ncp**2 / (2 * torch.clamp(degrees_of_freedom, min=1.0)),
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
