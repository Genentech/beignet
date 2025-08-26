import math

import torch
from torch import Tensor

from ._chi_squared_independence_power import chi_square_independence_power


def chi_square_independence_minimum_detectable_effect(
    sample_size: Tensor,
    rows: Tensor,
    cols: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Minimum detectable Cohen's w for chi-square independence tests.

    Parameters
    ----------
    sample_size : Tensor
        Total sample size.
    rows : Tensor
        Number of rows in the contingency table.
    cols : Tensor
        Number of columns in the contingency table.
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

    - **Contingency table analysis:** Testing association between two categorical variables
    - **Cross-tabulation studies:** Examining relationships between demographic factors
    - **Survey data analysis:** Testing independence of responses across categories
    - **Medical research:** Analyzing treatment outcomes by patient characteristics
    - **Quality control:** Testing independence between defect types and production factors
    - **Market research:** Examining relationships between consumer segments and preferences

    **Machine Learning Applications:**

    - **Feature independence testing:** Minimum detectable associations between categorical features
    - **Bias detection:** Smallest discriminatory patterns detectable in algorithmic decisions
    - **A/B testing segmentation:** Detecting treatment effects across different user segments
    - **User behavior analysis:** Minimum detectable correlations between user attributes and actions
    - **Content recommendation validation:** Testing independence between content types and user engagement
    - **Fraud detection pattern analysis:** Detecting associations between transaction attributes
    - **Classification model validation:** Testing predicted class independence from sensitive attributes
    - **Data preprocessing validation:** Ensuring categorical transformations maintain independence
    - **Clustering evaluation:** Testing independence between cluster assignments and external variables
    - **Multi-target prediction:** Validating independence assumptions between target variables
    - **Feature selection validation:** Testing categorical feature relationships in high-dimensional data
    - **Fairness testing in ML:** Detecting biased associations between protected and predicted classes
    - **Synthetic data evaluation:** Validating independence preservation in generated categorical data
    - **Model calibration across segments:** Testing prediction consistency across categorical groups
    - **Data quality assessment:** Detecting unexpected categorical variable dependencies

    **Interpretation Guidelines:**

    - **Cohen's w = 0.1:** Small association between variables
    - **Cohen's w = 0.3:** Medium association strength
    - **Cohen's w = 0.5:** Large association effect
    - **Values < 0.1:** Very weak associations, may lack practical significance
    - **Higher dimensions require larger samples:** More rows/columns reduce power
    - **Consider practical impact:** Statistical association doesn't guarantee causal relationship
    """
    sample_size_0 = torch.as_tensor(sample_size)
    r0 = torch.as_tensor(rows)
    c0 = torch.as_tensor(cols)
    scalar_out = sample_size_0.ndim == 0 and r0.ndim == 0 and c0.ndim == 0
    sample_size = torch.atleast_1d(sample_size_0)
    r = torch.atleast_1d(r0)
    c = torch.atleast_1d(c0)
    dtype = (
        torch.float64
        if any(t.dtype == torch.float64 for t in (sample_size, r, c))
        else torch.float32
    )
    sample_size = torch.clamp(sample_size.to(dtype), min=1.0)
    r = torch.clamp(r.to(dtype), min=2.0)
    c = torch.clamp(c.to(dtype), min=2.0)

    # Initial guess
    sqrt2 = math.sqrt(2.0)
    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2
    w0 = torch.clamp((z_alpha + z_beta) / torch.sqrt(sample_size), min=1e-8)

    w_lo = torch.zeros_like(w0) + 1e-8
    w_hi = torch.clamp(2.0 * w0 + 1e-6, min=1e-6)

    for _ in range(8):
        p_hi = chi_square_independence_power(w_hi, sample_size, r, c, alpha)
        need_expand = p_hi < power
        if not torch.any(need_expand):
            break
        w_hi = torch.where(need_expand, w_hi * 2.0, w_hi)
        w_hi = torch.clamp(w_hi, max=torch.tensor(10.0, dtype=dtype))

    w = (w_lo + w_hi) * 0.5
    for _ in range(24):
        p_mid = chi_square_independence_power(w, sample_size, r, c, alpha)
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
