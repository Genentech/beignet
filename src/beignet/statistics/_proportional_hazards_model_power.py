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

    When to Use
    -----------
    **Traditional Statistics:**
    - Survival analysis in clinical trials (time to death, disease progression)
    - Reliability engineering (time to equipment failure)
    - Customer analytics (time to churn, purchase)
    - Medical device studies (time to device failure)
    - Longitudinal studies with censored outcomes

    **Machine Learning Contexts:**
    - Churn prediction model validation with time-to-event outcomes
    - Predictive maintenance: validating time-to-failure models
    - A/B testing with time-to-conversion metrics
    - Recommendation systems: time until user engagement
    - Fraud detection: time until fraudulent activity
    - Clinical AI: survival prediction model validation
    - Marketing analytics: customer lifetime value modeling
    - Software reliability: time between software failures
    - IoT applications: sensor failure time prediction

    **Choose Cox regression power when:**
    - Outcome is time-to-event (survival time)
    - Presence of censored observations (incomplete follow-up)
    - Proportional hazards assumption holds
    - Interest in hazard ratios rather than absolute survival times
    - Multiple covariates affecting survival

    **Hazard Ratio Interpretation:**
    - HR = 1: no effect
    - HR > 1: increased hazard (worse survival, faster failure)
    - HR < 1: decreased hazard (better survival, delayed failure)
    - HR = 0.5: 50% reduction in hazard
    - HR = 2.0: doubling of hazard

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
    >>> hazard_ratio = torch.tensor(0.7)
    >>> n_events = torch.tensor(100)
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

    dtypes = [hazard_ratio.dtype, n_events.dtype, p_exposed.dtype]
    if any(dt == torch.float64 for dt in dtypes):
        dtype = torch.float64
    else:
        dtype = torch.float32

    hazard_ratio = hazard_ratio.to(dtype)
    n_events = n_events.to(dtype)
    p_exposed = p_exposed.to(dtype)

    hazard_ratio = torch.clamp(hazard_ratio, min=0.01, max=100.0)
    n_events = torch.clamp(n_events, min=5.0)
    p_exposed = torch.clamp(p_exposed, min=0.01, max=0.99)

    log_hr = torch.log(hazard_ratio)

    variance_null = n_events * p_exposed * (1.0 - p_exposed)

    ncp = torch.sqrt(variance_null) * torch.abs(log_hr)

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
        power = 0.5 * (1 - torch.erf((z_alpha - ncp) / sqrt2)) + 0.5 * (
            1 - torch.erf((z_alpha + ncp) / sqrt2)
        )
    elif alt == "greater":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
        if torch.all(hazard_ratio >= 1.0):
            power = 0.5 * (1 - torch.erf((z_alpha - ncp) / sqrt2))
        else:
            power = 0.5 * (1 - torch.erf((z_alpha - ncp) / sqrt2)) + 0.5 * (
                1 - torch.erf((z_alpha + ncp) / sqrt2)
            )
    else:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
        if torch.all(hazard_ratio <= 1.0):
            power = 0.5 * (1 - torch.erf((z_alpha + ncp) / sqrt2))
        else:
            power = 0.5 * (1 - torch.erf((z_alpha - ncp) / sqrt2)) + 0.5 * (
                1 - torch.erf((z_alpha + ncp) / sqrt2)
            )

    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)
        return out
    return power
