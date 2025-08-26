import math

import torch
from torch import Tensor

from ._interrupted_time_series_power import interrupted_time_series_power


def interrupted_time_series_sample_size(
    effect_size: Tensor,
    pre_post_ratio: Tensor = 1.0,
    autocorrelation: Tensor = 0.0,
    power: float = 0.8,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Required number of time points for interrupted time series analysis.

    Calculates the total number of time points needed to achieve desired power
    for detecting an intervention effect in interrupted time series analysis.

    When to Use
    -----------
    **Traditional Statistics:**
    - Planning policy evaluation studies with adequate temporal resolution
    - Designing clinical intervention studies with time series outcomes
    - Public health program evaluation planning with population-level metrics
    - Educational intervention planning with longitudinal assessment data
    - Economic policy analysis planning with sufficient temporal coverage

    **Machine Learning Contexts:**
    - A/B testing planning: determining required observation periods for temporal experiments
    - Model deployment studies: planning sufficient time points for performance impact assessment
    - Feature engineering experiments: determining temporal coverage for feature impact evaluation
    - Concept drift studies: planning adequate time points for distribution change detection
    - Online learning evaluation: determining required observation periods for algorithm assessment
    - Recommendation system experiments: planning adequate time for user behavior change detection
    - Anomaly detection validation: determining required time points for system effectiveness evaluation
    - Causal inference planning: ensuring adequate temporal coverage for treatment effect detection
    - Reinforcement learning studies: planning adequate episodes for policy change evaluation
    - Time series forecasting: planning sufficient pre/post observations for model comparison

    **Use this function when:**
    - Planning time series experiments with clear intervention points
    - Expected effect size can be reasonably estimated from pilot studies
    - Data collection frequency and duration are controllable
    - Interest is in detecting immediate level changes (not gradual trends)
    - Temporal autocorrelation can be estimated or assumed

    **Sample size considerations:**
    - Larger effect sizes require fewer time points
    - Higher autocorrelation generally requires more time points
    - Balanced pre/post periods (ratio = 1.0) typically provide optimal power
    - Seasonal data may require specific ratios to capture full cycles
    - Cost-effectiveness may favor shorter pre-intervention periods

    Parameters
    ----------
    effect_size : Tensor
        Standardized effect size for the intervention effect (Cohen's d).
    pre_post_ratio : Tensor, default=1.0
        Ratio of pre-intervention to post-intervention time points.
        A ratio of 1.0 means equal periods before and after intervention.
    autocorrelation : Tensor, default=0.0
        First-order autocorrelation (AR1 parameter). Range [-1, 1].
    power : float, default=0.8
        Desired statistical power.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    output : Tensor
        Required total number of time points (rounded up to nearest integer).

    Examples
    --------
    >>> effect_size = torch.tensor(0.8)  # Large effect
    >>> interrupted_time_series_sample_size(effect_size)
    tensor(30.0)

    Notes
    -----
    This function uses an iterative approach to find the total number of
    time points that achieves the desired power.

    The pre_post_ratio determines how time points are allocated:
    - ratio = 1.0: equal pre and post periods (optimal for power)
    - ratio > 1.0: more pre-intervention points
    - ratio < 1.0: more post-intervention points

    Time series design considerations:
    - Minimum 6 time points total (3 before, 3 after intervention)
    - Balanced designs (ratio ≈ 1) typically provide optimal power
    - Higher autocorrelation reduces effective sample size
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    pre_post_ratio = torch.atleast_1d(torch.as_tensor(pre_post_ratio))
    autocorrelation = torch.atleast_1d(torch.as_tensor(autocorrelation))

    # Ensure floating point dtype
    dtypes = [effect_size.dtype, pre_post_ratio.dtype, autocorrelation.dtype]
    if any(dt == torch.float64 for dt in dtypes):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    pre_post_ratio = pre_post_ratio.to(dtype)
    autocorrelation = autocorrelation.to(dtype)

    # Validate inputs
    effect_size = torch.clamp(effect_size, min=1e-8)
    pre_post_ratio = torch.clamp(pre_post_ratio, min=0.1, max=10.0)
    autocorrelation = torch.clamp(autocorrelation, min=-0.99, max=0.99)

    # Initial approximation
    sqrt2 = math.sqrt(2.0)
    z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt2
    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2

    # Autocorrelation adjustment
    if torch.any(torch.abs(autocorrelation) > 1e-6):
        ar_adjustment = (1.0 - autocorrelation**2) / (1.0 + autocorrelation**2)
    else:
        ar_adjustment = torch.ones_like(autocorrelation)

    # Design efficiency: optimal when pre_post_ratio ≈ 1
    # p_post = 1/(1 + pre_post_ratio), design_var = p*(1-p)
    p_post = 1.0 / (1.0 + pre_post_ratio)
    design_variance = p_post * (1.0 - p_post)

    # Initial sample size approximation
    n_init = ((z_alpha + z_beta) / effect_size) ** 2
    n_init = n_init / (ar_adjustment * design_variance)
    n_init = torch.clamp(n_init, min=10.0)

    # Calculate pre/post split
    n_pre_init = torch.ceil(n_init * pre_post_ratio / (1.0 + pre_post_ratio))
    n_pre_init = torch.clamp(n_pre_init, min=3.0)

    # Iterative refinement
    n_total_current = n_init
    for _ in range(15):
        # Calculate current pre/post split
        n_pre_current = torch.ceil(
            n_total_current * pre_post_ratio / (1.0 + pre_post_ratio)
        )
        n_pre_current = torch.clamp(
            n_pre_current, min=torch.tensor(3.0, dtype=dtype), max=n_total_current - 3.0
        )

        # Calculate current power
        current_power = interrupted_time_series_power(
            effect_size, n_total_current, n_pre_current, autocorrelation, alpha=alpha
        )

        # Adjust sample size based on power gap
        power_gap = torch.clamp(power - current_power, min=-0.4, max=0.4)
        adjustment = 1.0 + 1.4 * power_gap
        n_total_current = torch.clamp(n_total_current * adjustment, min=10.0, max=1e4)

    # Round up to nearest integer
    n_out = torch.ceil(n_total_current)

    if out is not None:
        out.copy_(n_out)
        return out
    return n_out
