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
    >>> effect_size = torch.tensor(0.8)  # Large effect
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

    # Ensure floating point dtype
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

    # Validate inputs
    effect_size = torch.clamp(effect_size, min=0.0)
    n_time_points = torch.clamp(n_time_points, min=6.0)
    n_pre_intervention = torch.clamp(
        n_pre_intervention, min=3.0, max=n_time_points - 3.0
    )
    autocorrelation = torch.clamp(autocorrelation, min=-0.99, max=0.99)

    n_post_intervention = n_time_points - n_pre_intervention

    # Effective sample size adjustment for autocorrelation
    # Approximate variance inflation factor
    if torch.any(torch.abs(autocorrelation) > 1e-6):
        # AR(1) adjustment: effective n ≈ n * (1-ρ²) / (1+ρ²) for balanced design
        ar_adjustment = (1.0 - autocorrelation**2) / (1.0 + autocorrelation**2)
    else:
        ar_adjustment = torch.ones_like(autocorrelation)

    effective_n = n_time_points * ar_adjustment

    # Design matrix considerations for segmented regression
    # The intervention indicator has variance p*(1-p) where p = n_post/n_total
    p_post = n_post_intervention / n_time_points
    design_variance = p_post * (1.0 - p_post)

    # Standard error for the intervention effect
    # SE(β₂) ≈ σ / √(n * Var(I_t))
    se_intervention = 1.0 / torch.sqrt(effective_n * design_variance)

    # Noncentrality parameter
    ncp = effect_size / se_intervention

    # Degrees of freedom (approximate)
    # df = n - p_parameters, where p_parameters ≈ 4 for basic ITS model
    df_approx = torch.clamp(effective_n - 4.0, min=1.0)

    # Critical value (two-sided test)
    sqrt2 = math.sqrt(2.0)
    z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt2

    # Adjust for finite degrees of freedom
    t_critical = z_alpha * torch.sqrt(1.0 + 2.0 / df_approx)

    # Power calculation
    z_score = t_critical - ncp
    power = 0.5 * (1 - torch.erf(z_score / sqrt2))

    # Clamp to valid range
    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)
        return out
    return power
