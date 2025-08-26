import math

import torch
from torch import Tensor

from ._chi_squared_goodness_of_fit_power import chi_square_goodness_of_fit_power


def chi_square_goodness_of_fit_minimum_detectable_effect(
    sample_size: Tensor,
    df: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Minimum detectable Cohen's w for chi-square goodness-of-fit tests.

    Parameters
    ----------
    sample_size : Tensor
        Total sample size.
    df : Tensor
        Degrees of freedom (categories - 1).
    power : float, default=0.8
        Target power.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    Tensor
        Minimal Cohen's w achieving the requested power.

    When to Use
    -----------
    **Traditional Statistics:**

    - **Distribution testing:** Determining if observed frequencies match expected theoretical distributions
    - **Model validation:** Testing goodness-of-fit for statistical models against data
    - **Categorical data analysis:** When expected frequencies for each category are known
    - **Quality control:** Testing if production distributions match specifications
    - **Survey research:** Validating response patterns against expected distributions
    - **Market research:** Testing consumer preference distributions against theoretical models

    **Machine Learning Applications:**

    - **Class imbalance detection:** Minimum detectable deviations from expected class distributions
    - **Data drift monitoring:** Smallest distribution shifts detectable in production data
    - **Feature distribution validation:** Testing if features match expected categorical distributions
    - **Synthetic data quality assessment:** Validating generated data against target distributions
    - **A/B testing categorical outcomes:** Minimum detectable changes in categorical response patterns
    - **Recommendation system evaluation:** Distribution differences in user interaction patterns
    - **Text classification validation:** Testing predicted category distributions against benchmarks
    - **Clustering evaluation:** Validating cluster membership distributions against expected patterns
    - **Fraud detection calibration:** Testing alert distribution patterns against baseline expectations
    - **User behavior analysis:** Detecting changes in categorical user action distributions
    - **Content performance testing:** Minimum detectable shifts in content category engagement
    - **Algorithm fairness validation:** Testing for demographic distribution deviations
    - **Model calibration assessment:** Validating predicted probability distributions
    - **Feature importance validation:** Testing categorical feature contribution patterns
    - **Data quality monitoring:** Detecting categorical data corruption or processing errors

    **Interpretation Guidelines:**

    - **Cohen's w = 0.1:** Small effect (small deviation from expected distribution)
    - **Cohen's w = 0.3:** Medium effect (moderate distribution deviation)
    - **Cohen's w = 0.5:** Large effect (substantial distribution difference)
    - **Values < 0.1:** Very small deviations, may lack practical significance
    - **Higher df requires larger sample sizes:** More categories reduce power for given effect size
    - **Consider practical impact:** Statistical significance doesn't guarantee business relevance
    """
    sample_size_0 = torch.as_tensor(sample_size)
    degrees_of_freedom_0 = torch.as_tensor(df)
    scalar_out = sample_size_0.ndim == 0 and degrees_of_freedom_0.ndim == 0
    sample_size = torch.atleast_1d(sample_size_0)
    degrees_of_freedom = torch.atleast_1d(degrees_of_freedom_0)
    dtype = (
        torch.float64
        if (
            sample_size.dtype == torch.float64
            or degrees_of_freedom.dtype == torch.float64
        )
        else torch.float32
    )
    sample_size = torch.clamp(sample_size.to(dtype), min=1.0)
    degrees_of_freedom = torch.clamp(degrees_of_freedom.to(dtype), min=1.0)

    sqrt2 = math.sqrt(2.0)
    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2
    w0 = torch.clamp((z_alpha + z_beta) / torch.sqrt(sample_size), min=1e-8)

    w_lo = torch.zeros_like(w0) + 1e-8
    w_hi = torch.clamp(2.0 * w0 + 1e-6, min=1e-6)

    for _ in range(8):
        p_hi = chi_square_goodness_of_fit_power(
            w_hi, sample_size, degrees_of_freedom, alpha
        )
        need_expand = p_hi < power
        if not torch.any(need_expand):
            break
        w_hi = torch.where(need_expand, w_hi * 2.0, w_hi)
        w_hi = torch.clamp(w_hi, max=torch.tensor(10.0, dtype=dtype))

    w = (w_lo + w_hi) * 0.5
    for _ in range(24):
        p_mid = chi_square_goodness_of_fit_power(
            w, sample_size, degrees_of_freedom, alpha
        )
        go_right = p_mid < power
        w_lo = torch.where(go_right, w, w_lo)
        w_hi = torch.where(go_right, w_hi, w)
        w = (w_lo + w_hi) * 0.5

    out_t = torch.clamp(w, min=0.0)
    if scalar_out:
        out_scalar = out_t.reshape(())
        if out is not None:
            out.copy_(out_scalar)
            return out
        return out_scalar
    else:
        if out is not None:
            out.copy_(out_t)
            return out
        return out_t
