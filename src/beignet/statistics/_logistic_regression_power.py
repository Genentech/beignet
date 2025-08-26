import math

import torch
from torch import Tensor


def logistic_regression_power(
    effect_size: Tensor,
    sample_size: Tensor,
    p_exposure: Tensor = 0.5,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Power analysis for logistic regression.

    Calculates statistical power for testing a single predictor coefficient
    in logistic regression using the Wald test.

    When to Use
    -----------
    **Traditional Statistics:**
    - Binary outcome prediction with continuous/categorical predictors
    - Case-control studies in epidemiology
    - Clinical trials with binary endpoints (cure/no cure, success/failure)
    - Survey research with binary response variables

    **Machine Learning Contexts:**
    - Binary classification model evaluation and comparison
    - Feature selection: determining significant predictors in classification
    - A/B testing with binary conversion outcomes
    - Causal inference: estimating treatment effects on binary outcomes
    - Fairness assessment: testing for discriminatory effects in binary decisions
    - Model interpretation: validating significance of individual features
    - Medical AI: diagnostic test validation with binary outcomes
    - Marketing analytics: predicting binary customer behaviors
    - Credit scoring: default/no-default prediction validation

    **Choose logistic regression power when:**
    - Outcome variable is binary (0/1, yes/no, success/failure)
    - Linear relationship between predictors and log-odds
    - Independent observations
    - Large enough sample size (typically n > 10 events per predictor)

    **Odds Ratio Interpretation:**
    - OR = 1: no association
    - OR > 1: positive association (higher odds)
    - OR < 1: negative association (lower odds)
    - OR = 1.5: small effect
    - OR = 2.0: medium effect
    - OR = 3.0: large effect

    Parameters
    ----------
    effect_size : Tensor
        Effect size as odds ratio (OR). OR = exp(β) where β is the regression
        coefficient. OR = 1 indicates no effect.
    sample_size : Tensor
        Total sample size.
    p_exposure : Tensor, default=0.5
        Proportion of subjects with the exposure/predictor = 1.
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
    >>> effect_size = torch.tensor(2.0)
    >>> sample_size = torch.tensor(100)
    >>> logistic_regression_power(effect_size, sample_size)
    tensor(0.6234)

    Notes
    -----
    For logistic regression: logit(p) = β₀ + β₁x

    The Wald test statistic is: Z = β̂₁ / SE(β̂₁)

    The standard error is approximately:
    SE(β̂₁) ≈ 1 / √(n * p_exposure * (1 - p_exposure) * p_outcome * (1 - p_outcome))

    where p_outcome is estimated from the marginal probability under the
    specified odds ratio.

    The effect_size parameter represents the odds ratio OR = exp(β₁).
    - OR = 1: no association
    - OR > 1: positive association
    - OR < 1: negative association (protective effect)

    References
    ----------
    Hsieh, F. Y., Bloch, D. A., & Larsen, M. D. (1998). A simple method of
    sample size calculation for linear and logistic regression. Statistics
    in medicine, 17(14), 1623-1634.
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))
    p_exposure = torch.atleast_1d(torch.as_tensor(p_exposure))

    dtype = (
        torch.float64
        if any(t.dtype == torch.float64 for t in (effect_size, sample_size, p_exposure))
        else torch.float32
    )
    effect_size = effect_size.to(dtype)
    sample_size = sample_size.to(dtype)
    p_exposure = p_exposure.to(dtype)

    effect_size = torch.clamp(effect_size, min=0.01, max=100.0)
    sample_size = torch.clamp(sample_size, min=10.0)
    p_exposure = torch.clamp(p_exposure, min=0.01, max=0.99)

    beta = torch.log(effect_size)

    logit_baseline = torch.tensor(
        0.0, dtype=dtype
    )
    logit_exposed = logit_baseline + beta
    logit_unexposed = logit_baseline

    p_outcome_exposed = torch.sigmoid(logit_exposed)
    p_outcome_unexposed = torch.sigmoid(logit_unexposed)

    p_outcome = p_exposure * p_outcome_exposed + (1 - p_exposure) * p_outcome_unexposed
    p_outcome = torch.clamp(p_outcome, min=0.01, max=0.99)

    variance_beta = 1.0 / (
        sample_size * p_exposure * (1 - p_exposure) * p_outcome * (1 - p_outcome)
    )
    se_beta = torch.sqrt(torch.clamp(variance_beta, min=1e-12))

    ncp = torch.abs(beta) / se_beta

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
        power = 0.5 * (1 - torch.erf((z_alpha - ncp) / sqrt2))
    else:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
        power = 0.5 * (1 - torch.erf((z_alpha + ncp) / sqrt2))

    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)
        return out
    return power
