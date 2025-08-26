import math

import torch
from torch import Tensor

from ._logistic_regression_power import logistic_regression_power


def logistic_regression_sample_size(
    effect_size: Tensor,
    p_exposure: Tensor = 0.5,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Required sample size for logistic regression.

    Calculates the total sample size needed to achieve desired power for
    testing a single predictor coefficient in logistic regression.

    Parameters
    ----------
    effect_size : Tensor
        Effect size as odds ratio (OR). OR = exp(β) where β is the regression
        coefficient. OR = 1 indicates no effect.
    p_exposure : Tensor, default=0.5
        Proportion of subjects with the exposure/predictor = 1.
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
    >>> effect_size = torch.tensor(2.0)
    >>> logistic_regression_sample_size(effect_size)
    tensor(194.0)

    Notes
    -----
    This function uses the Hsieh et al. (1998) approximation for the variance
    of the logistic regression coefficient, combined with an iterative approach
    to find the sample size that achieves the desired power.

    The sample size depends on:
    - Effect size (odds ratio)
    - Exposure prevalence (p_exposure)
    - Outcome prevalence (estimated from effect size)
    - Desired power and significance level

    For optimal power, p_exposure should be close to 0.5 (balanced design).
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    p_exposure = torch.atleast_1d(torch.as_tensor(p_exposure))

    dtype = (
        torch.float64
        if (effect_size.dtype == torch.float64 or p_exposure.dtype == torch.float64)
        else torch.float32
    )
    effect_size = effect_size.to(dtype)
    p_exposure = p_exposure.to(dtype)

    effect_size = torch.clamp(effect_size, min=0.01, max=100.0)
    p_exposure = torch.clamp(p_exposure, min=0.01, max=0.99)

    beta = torch.log(effect_size)

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

    p_outcome_approx = torch.tensor(0.5, dtype=dtype)
    variance_approx = 1.0 / (
        p_exposure * (1 - p_exposure) * p_outcome_approx * (1 - p_outcome_approx)
    )

    n_init = ((z_alpha + z_beta) ** 2) * variance_approx / (beta**2)
    n_init = torch.clamp(n_init, min=20.0)

    n_current = n_init
    for _ in range(15):
        current_power = logistic_regression_power(
            effect_size, n_current, p_exposure, alpha=alpha, alternative=alternative
        )

        power_gap = torch.clamp(power - current_power, min=-0.4, max=0.4)
        adjustment = 1.0 + 1.2 * power_gap
        n_current = torch.clamp(n_current * adjustment, min=20.0, max=1e6)

    n_out = torch.ceil(n_current)

    if out is not None:
        out.copy_(n_out)
        return out
    return n_out
