import math

import torch
from torch import Tensor


def proportional_hazards_model_power(
    hazard_ratio: Tensor,
    n_events: Tensor,
    p_exposed: Tensor = 0.5,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Power analysis for Cox proportional hazards model.

    Calculates statistical power for testing a single covariate coefficient
    in the Cox proportional hazards model using the log-rank test or
    Wald test approximation.

    Parameters
    ----------
    hazard_ratio : Tensor
        Expected hazard ratio (HR) under the alternative hypothesis.
        HR = exp(β) where β is the regression coefficient.
        HR = 1 indicates no effect.
    n_events : Tensor
        Expected number of events (deaths, failures) during the study.
    p_exposed : Tensor, default=0.5
        Proportion of subjects in the exposed/treatment group.
        Should be in range (0, 1).
    alpha : float, default=0.05
        Significance level.
    alternative : {"two-sided", "greater", "less"}, default="two-sided"
        Alternative hypothesis direction.

    Returns
    -------
    output : Tensor, shape=(...)
        Statistical power values in range [0, 1].

    Examples
    --------
    >>> hazard_ratio = torch.tensor(0.7)  # 30% reduction in hazard
    >>> n_events = torch.tensor(100)      # 100 events expected
    >>> proportional_hazards_model_power(hazard_ratio, n_events)
    tensor(0.7234)

    Notes
    -----
    Cox proportional hazards model: h(t|x) = h₀(t) × exp(βx)

    Where:
    - h(t|x) is the hazard at time t given covariate x
    - h₀(t) is the baseline hazard
    - β is the regression coefficient
    - HR = exp(β)

    The log-rank test statistic is approximately:
    Z = (O₁ - E₁) / √V

    Where:
    - O₁ = observed events in group 1
    - E₁ = expected events in group 1 under null
    - V = variance of O₁ - E₁

    Under the alternative hypothesis with hazard ratio HR:
    Z ~ N(√(n_events × p_exposed × (1-p_exposed) × ln²(HR)), 1)

    Effect interpretation:
    - HR = 1: no effect
    - HR > 1: increased hazard (worse prognosis)
    - HR < 1: decreased hazard (better prognosis)

    References
    ----------
    Schoenfeld, D. A. (1983). Sample-size formula for the proportional-hazards
    regression model. Biometrics, 499-503.
    """
    hazard_ratio = torch.atleast_1d(torch.as_tensor(hazard_ratio))
    n_events = torch.atleast_1d(torch.as_tensor(n_events))
    p_exposed = torch.atleast_1d(torch.as_tensor(p_exposed))

    # Ensure floating point dtype
    dtypes = [hazard_ratio.dtype, n_events.dtype, p_exposed.dtype]
    if any(dt == torch.float64 for dt in dtypes):
        dtype = torch.float64
    else:
        dtype = torch.float32

    hazard_ratio = hazard_ratio.to(dtype)
    n_events = n_events.to(dtype)
    p_exposed = p_exposed.to(dtype)

    # Validate inputs
    hazard_ratio = torch.clamp(hazard_ratio, min=0.01, max=100.0)
    n_events = torch.clamp(n_events, min=5.0)
    p_exposed = torch.clamp(p_exposed, min=0.01, max=0.99)

    # Convert HR to log coefficient
    log_hr = torch.log(hazard_ratio)

    # Variance of log-rank test statistic under null hypothesis
    # V = n_events × p_exposed × (1 - p_exposed)
    variance_null = n_events * p_exposed * (1.0 - p_exposed)

    # Noncentrality parameter (expected Z under alternative)
    # δ = √(V) × ln(HR) = √(n_events × p × (1-p)) × ln(HR)
    ncp = torch.sqrt(variance_null) * torch.abs(log_hr)

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
        # Two-sided power: P(|Z| > z_α/2 | H₁)
        power = 0.5 * (1 - torch.erf((z_alpha - ncp) / sqrt2)) + 0.5 * (
            1 - torch.erf((z_alpha + ncp) / sqrt2)
        )
    elif alt == "greater":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
        # One-sided power: P(Z > z_α | H₁)
        # For HR > 1 (increased hazard)
        if torch.all(hazard_ratio >= 1.0):
            power = 0.5 * (1 - torch.erf((z_alpha - ncp) / sqrt2))
        else:
            # Mixed case: use two-sided approximation
            power = 0.5 * (1 - torch.erf((z_alpha - ncp) / sqrt2)) + 0.5 * (
                1 - torch.erf((z_alpha + ncp) / sqrt2)
            )
    else:  # alt == "less"
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
        # One-sided power: P(Z < -z_α | H₁)
        # For HR < 1 (decreased hazard)
        if torch.all(hazard_ratio <= 1.0):
            power = 0.5 * (1 - torch.erf((z_alpha + ncp) / sqrt2))
        else:
            # Mixed case: use two-sided approximation
            power = 0.5 * (1 - torch.erf((z_alpha - ncp) / sqrt2)) + 0.5 * (
                1 - torch.erf((z_alpha + ncp) / sqrt2)
            )

    # Clamp to valid range
    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)
        return out
    return power
