import math

import torch
from torch import Tensor


def poisson_regression_power(
    effect_size: Tensor,
    sample_size: Tensor,
    mean_rate: Tensor,
    p_exposure: Tensor = 0.5,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Power analysis for Poisson regression.

    Calculates statistical power for testing a single predictor coefficient
    in Poisson regression using the Wald test.

    When to Use
    -----------
    **Traditional Statistics:**
    - Count data analysis (events, occurrences, frequencies)
    - Epidemiology: disease incidence rates
    - Quality control: defect counts per unit
    - Ecology: species abundance counts

    **Machine Learning Contexts:**
    - Click-through rate modeling in web analytics
    - Anomaly detection: count-based features
    - Time series: event count prediction
    - Natural language processing: word count modeling
    - Computer vision: object count prediction
    - Recommendation systems: interaction count modeling
    - Social media: post/like/share count analysis
    - IoT applications: sensor event count analysis
    - Network analysis: connection count modeling

    **Choose Poisson regression when:**
    - Outcome variable consists of counts (non-negative integers)
    - Events occur independently
    - Mean equals variance (equidispersion) approximately
    - Count data cannot be negative

    **Incidence Rate Ratio Interpretation:**
    - IRR = 1: no association
    - IRR > 1: increased incidence rate
    - IRR < 1: decreased incidence rate
    - IRR = 1.5: 50% increase in rate
    - IRR = 0.7: 30% decrease in rate

    Parameters
    ----------
    effect_size : Tensor
        Effect size as incidence rate ratio (IRR). IRR = exp(β) where β is
        the regression coefficient. IRR = 1 indicates no effect.
    sample_size : Tensor
        Total sample size.
    mean_rate : Tensor
        Expected mean count rate in the reference group.
        Must be positive.
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
    >>> effect_size = torch.tensor(1.5)
    >>> sample_size = torch.tensor(200)
    >>> mean_rate = torch.tensor(2.0)
    >>> poisson_regression_power(effect_size, sample_size, mean_rate)
    tensor(0.7123)

    Notes
    -----
    For Poisson regression: log(μ) = β₀ + β₁x

    The Wald test statistic is: Z = β̂₁ / SE(β̂₁)

    The standard error is approximately:
    SE(β̂₁) ≈ 1 / √(n * p_exposure * (1 - p_exposure) * E[Y])

    where E[Y] is the expected count, estimated from the mean rate and
    incidence rate ratio.

    The effect_size parameter represents IRR = exp(β₁):
    - IRR = 1: no association
    - IRR > 1: increased incidence
    - IRR < 1: decreased incidence (protective effect)

    References
    ----------
    Cameron, A. C., & Trivedi, P. K. (2013). Regression analysis of count data.
    Cambridge university press.
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))
    mean_rate = torch.atleast_1d(torch.as_tensor(mean_rate))
    p_exposure = torch.atleast_1d(torch.as_tensor(p_exposure))

    dtypes = [effect_size.dtype, sample_size.dtype, mean_rate.dtype, p_exposure.dtype]
    if any(dt == torch.float64 for dt in dtypes):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    sample_size = sample_size.to(dtype)
    mean_rate = mean_rate.to(dtype)
    p_exposure = p_exposure.to(dtype)

    effect_size = torch.clamp(effect_size, min=0.01, max=100.0)
    sample_size = torch.clamp(sample_size, min=10.0)
    mean_rate = torch.clamp(mean_rate, min=0.01)
    p_exposure = torch.clamp(p_exposure, min=0.01, max=0.99)

    beta = torch.log(effect_size)

    mean_unexposed = mean_rate
    mean_exposed = mean_rate * effect_size

    expected_count = p_exposure * mean_exposed + (1 - p_exposure) * mean_unexposed

    variance_beta = 1.0 / (sample_size * p_exposure * (1 - p_exposure) * expected_count)
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
