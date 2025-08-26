import math

import torch
from torch import Tensor

from ._proportional_hazards_model_power import proportional_hazards_model_power


def proportional_hazards_model_sample_size(
    hazard_ratio: Tensor,
    event_rate: Tensor,
    p_exposed: Tensor = 0.5,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Required sample size for Cox proportional hazards model.

    Calculates the total sample size needed to achieve desired power
    for detecting a hazard ratio in survival analysis. The calculation
    accounts for the expected event rate.

    Parameters
    ----------
    hazard_ratio : Tensor
        Expected hazard ratio (HR) under the alternative hypothesis.
        HR = exp(β) where β is the regression coefficient.
    event_rate : Tensor
        Expected proportion of subjects who will experience the event
        during the study period. Should be in range (0, 1).
    p_exposed : Tensor, default=0.5
        Proportion of subjects in the exposed/treatment group.
        Should be in range (0, 1).
    power : float, default=0.8
        Desired statistical power.
    alpha : float, default=0.05
        Significance level.
    alternative : {"two-sided", "greater", "less"}, default="two-sided"
        Alternative hypothesis direction.

    Returns
    -------
    output : Tensor
        Required total sample size (rounded up to nearest integer).

    Examples
    --------
    >>> hazard_ratio = torch.tensor(0.7)   # 30% reduction in hazard
    >>> event_rate = torch.tensor(0.6)     # 60% will have event
    >>> proportional_hazards_model_sample_size(hazard_ratio, event_rate)
    tensor(278.0)

    Notes
    -----
    This function calculates the total sample size needed, accounting for
    the fact that power depends on the number of events, not the total
    sample size directly.

    The relationship is:
    n_total = n_events / event_rate

    Sample size considerations:
    - Higher event rates require smaller total sample sizes
    - Balanced allocation (p_exposed ≈ 0.5) is optimal for power
    - The hazard ratio closer to 1 requires larger sample sizes
    - Longer follow-up increases event rate, reducing required sample size

    This assumes:
    - Proportional hazards assumption holds
    - No competing risks
    - Exponential or Weibull survival distribution
    """
    hazard_ratio = torch.atleast_1d(torch.as_tensor(hazard_ratio))
    event_rate = torch.atleast_1d(torch.as_tensor(event_rate))
    p_exposed = torch.atleast_1d(torch.as_tensor(p_exposed))

    # Ensure floating point dtype
    dtypes = [hazard_ratio.dtype, event_rate.dtype, p_exposed.dtype]
    if any(dt == torch.float64 for dt in dtypes):
        dtype = torch.float64
    else:
        dtype = torch.float32

    hazard_ratio = hazard_ratio.to(dtype)
    event_rate = event_rate.to(dtype)
    p_exposed = p_exposed.to(dtype)

    # Validate inputs
    hazard_ratio = torch.clamp(hazard_ratio, min=0.01, max=100.0)
    event_rate = torch.clamp(event_rate, min=0.01, max=0.99)
    p_exposed = torch.clamp(p_exposed, min=0.01, max=0.99)

    # Initial approximation for number of events needed
    log_hr = torch.log(hazard_ratio)

    # Critical values
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

    # Schoenfeld formula for number of events
    # n_events = (z_α + z_β)² / [p(1-p) × ln²(HR)]
    n_events_needed = ((z_alpha + z_beta) ** 2) / (
        p_exposed * (1.0 - p_exposed) * (log_hr**2)
    )
    n_events_needed = torch.clamp(n_events_needed, min=10.0)

    # Convert to total sample size
    n_total_init = n_events_needed / event_rate
    n_total_init = torch.clamp(n_total_init, min=20.0)

    # Iterative refinement using the exact power function
    n_current = n_total_init
    for _ in range(10):
        # Calculate expected number of events with current sample size
        expected_events = n_current * event_rate

        # Calculate current power
        current_power = proportional_hazards_model_power(
            hazard_ratio,
            expected_events,
            p_exposed,
            alpha=alpha,
            alternative=alternative,
        )

        # Adjust sample size based on power gap
        power_gap = torch.clamp(power - current_power, min=-0.4, max=0.4)
        adjustment = 1.0 + 1.1 * power_gap
        n_current = torch.clamp(n_current * adjustment, min=20.0, max=1e6)

    # Round up to nearest integer
    n_out = torch.ceil(n_current)

    if out is not None:
        out.copy_(n_out)
        return out
    return n_out
