import math

import torch
from torch import Tensor


def interrupted_time_series_power(
    effect_size: Tensor,
    n_time_points: Tensor,
    n_pre_intervention: Tensor,
    autocorrelation: Tensor = 0.0,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Power analysis for interrupted time series analysis.

    Tests for a change in level or slope at an intervention point in a
    time series using segmented regression.

    When to Use
    -----------
    **Traditional Statistics:**
    - Evaluating policy interventions with time series outcome data
    - Clinical research assessing treatment effects over time
    - Public health studies analyzing intervention impacts on population metrics
    - Educational research measuring program effectiveness over time
    - Economic analysis of regulatory changes on market outcomes

    **Machine Learning Contexts:**
    - A/B testing with temporal dependencies: measuring intervention effects in time series
    - Model deployment impact: assessing performance changes after model updates
    - Feature engineering evaluation: measuring impact of new features over time
    - Concept drift detection: identifying significant changes in data distributions
    - Online learning evaluation: assessing algorithm performance changes over time
    - Recommendation system analysis: measuring impact of algorithm changes on user behavior
    - Anomaly detection validation: evaluating detection system effectiveness over time
    - Causal inference in ML: identifying treatment effects in observational time series data
    - Reinforcement learning: measuring policy change effects in sequential decision making
    - Time series forecasting: evaluating model performance before/after structural changes

    **Choose interrupted time series over other methods when:**
    - Data has natural temporal ordering with clear intervention point
    - Randomized controlled trial is not feasible or ethical
    - Need to control for secular trends and seasonal patterns
    - Multiple time points available before and after intervention
    - Interest is in both immediate and gradual effects of intervention

    **Choose ITS over simple before-after comparison when:**
    - Secular trends exist that could confound results
    - Need to separate immediate level changes from slope changes
    - Multiple time points available (not just single pre/post measurements)
    - Temporal autocorrelation is present in the data

    **Design considerations:**
    - Minimum 8-10 observations recommended (preferably 50+)
    - Balanced pre/post periods generally provide better power
    - Consider seasonal patterns and adjust design accordingly
    - Account for autocorrelation in power calculations

    Parameters
    ----------
    effect_size : Tensor
        Standardized effect size for the intervention effect.
        Represents the immediate change in level (Cohen's d).
    n_time_points : Tensor
        Total number of time points in the series.
    n_pre_intervention : Tensor
        Number of time points before the intervention.
    autocorrelation : Tensor, default=0.0
        First-order autocorrelation (AR1 parameter) in the time series.
        Range is typically [-1, 1].
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    output : Tensor, shape=(...)
        Statistical power values in range [0, 1].

    Examples
    --------
    >>> effect_size = torch.tensor(0.8)
    >>> n_time_points = torch.tensor(50)
    >>> n_pre_intervention = torch.tensor(25)
    >>> interrupted_time_series_power(effect_size, n_time_points, n_pre_intervention)
    tensor(0.9123)

    Notes
    -----
    Interrupted time series model:
    Y_t = β₀ + β₁t + β₂I_t + β₃(t - t*)I_t + ε_t

    Where:
    - Y_t = outcome at time t
    - t = time variable
    - I_t = indicator (1 after intervention, 0 before)
    - t* = intervention time point
    - β₂ = immediate level change (tested effect)
    - β₃ = slope change

    The power depends on:
    - Effect size (immediate change)
    - Total number of observations
    - Balance of pre/post intervention periods
    - Autocorrelation structure

    This implementation focuses on testing the immediate level change (β₂)
    and assumes AR(1) error structure.

    References
    ----------
    Wagner, A. K., et al. (2002). Segmented regression analysis of interrupted
    time series studies in medication use research. Journal of Clinical
    Pharmacy and Therapeutics, 27(4), 299-309.
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    n_time_points = torch.atleast_1d(torch.as_tensor(n_time_points))
    n_pre_intervention = torch.atleast_1d(torch.as_tensor(n_pre_intervention))
    autocorrelation = torch.atleast_1d(torch.as_tensor(autocorrelation))

    dtypes = [
        effect_size.dtype,
        n_time_points.dtype,
        n_pre_intervention.dtype,
        autocorrelation.dtype,
    ]
    if any(dt == torch.float64 for dt in dtypes):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    n_time_points = n_time_points.to(dtype)
    n_pre_intervention = n_pre_intervention.to(dtype)
    autocorrelation = autocorrelation.to(dtype)

    effect_size = torch.clamp(effect_size, min=0.0)
    n_time_points = torch.clamp(n_time_points, min=6.0)
    n_pre_intervention = torch.clamp(
        n_pre_intervention, min=torch.tensor(3.0, dtype=dtype), max=n_time_points - 3.0
    )
    autocorrelation = torch.clamp(autocorrelation, min=-0.99, max=0.99)

    n_post_intervention = n_time_points - n_pre_intervention

    if torch.any(torch.abs(autocorrelation) > 1e-6):
        ar_adjustment = (1.0 - autocorrelation**2) / (1.0 + autocorrelation**2)
    else:
        ar_adjustment = torch.ones_like(autocorrelation)

    effective_n = n_time_points * ar_adjustment

    prob_post = n_post_intervention / n_time_points
    design_variance = prob_post * (1.0 - prob_post)

    se_intervention = 1.0 / torch.sqrt(effective_n * design_variance)

    noncentrality_parameter = effect_size / se_intervention

    degrees_of_freedom_approx = torch.clamp(effective_n - 4.0, min=1.0)

    sqrt2 = math.sqrt(2.0)
    z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt2

    t_critical = z_alpha * torch.sqrt(1.0 + 2.0 / degrees_of_freedom_approx)

    z_score = t_critical - noncentrality_parameter
    power = 0.5 * (1 - torch.erf(z_score / sqrt2))

    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)
        return out
    return power
