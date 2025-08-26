import math

import torch
from torch import Tensor


def correlation_minimum_detectable_effect(
    sample_size: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Minimum detectable absolute correlation |r| using Fisher z approximation.

    Parameters
    ----------
    sample_size : Tensor
        Total number of observations.
    power : float, default=0.8
        Target power.
    alpha : float, default=0.05
        Significance level.
    alternative : {"two-sided", "greater", "less"}
        Test direction. This function returns a non-negative magnitude |r| for
        two-sided; for one-sided it returns the magnitude in the specified direction.

    Returns
    -------
    Tensor
        Minimal |r| (non-negative; apply sign per alternative if needed).

    When to Use
    -----------
    **Traditional Statistics:**

    - **Correlation analysis:** Determining the smallest meaningful correlation detectable with available data
    - **Psychometric validation:** Minimum detectable correlations between test items or constructs
    - **Market research:** Smallest associations detectable between consumer attributes and behaviors
    - **Quality control:** Minimum detectable relationships between process variables and outcomes
    - **Survey research:** Planning studies to detect meaningful correlations between variables
    - **Medical research:** Minimum detectable correlations between biomarkers and clinical outcomes

    **Machine Learning Applications:**

    - **Feature correlation analysis:** Minimum detectable linear relationships between continuous features
    - **Model validation:** Smallest correlations detectable between predicted and actual values
    - **Feature selection threshold setting:** Determining correlation cutoffs for feature filtering
    - **Data quality assessment:** Detecting minimum meaningful relationships in data validation
    - **Time series analysis:** Minimum detectable correlations between temporal features
    - **Hyperparameter optimization:** Correlation thresholds for parameter relationship analysis
    - **Ensemble method evaluation:** Minimum detectable correlations between base model predictions
    - **Cross-validation consistency:** Testing correlation stability across validation folds
    - **Feature engineering validation:** Minimum detectable improvements from feature transformations
    - **Anomaly detection calibration:** Correlation thresholds for outlier identification
    - **Dimensionality reduction evaluation:** Minimum detectable correlations in reduced spaces
    - **Transfer learning assessment:** Correlation analysis between source and target domain features
    - **Multi-modal learning:** Minimum detectable correlations between different data modalities
    - **Causal inference preparation:** Correlation analysis as preliminary step to causal modeling
    - **Active learning strategies:** Correlation-based sample selection thresholds

    **Interpretation Guidelines:**

    - **|r| = 0.1:** Small correlation (1% shared variance)
    - **|r| = 0.3:** Medium correlation (9% shared variance)
    - **|r| = 0.5:** Large correlation (25% shared variance)
    - **Values < 0.1:** Very weak relationships, may lack practical significance
    - **Sample size critical:** Larger samples can detect smaller correlations reliably
    - **Consider non-linear relationships:** Pearson correlation only captures linear associations
    """
    n0 = torch.as_tensor(sample_size)
    scalar_out = n0.ndim == 0
    n = torch.atleast_1d(n0)
    dtype = torch.float64 if n.dtype == torch.float64 else torch.float32
    n = torch.clamp(n.to(dtype), min=4.0)  # need n>3 for Fisher z

    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    sqrt2 = math.sqrt(2.0)
    if alt == "two-sided":
        z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt2
    else:
        z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2

    # Fisher z: SE = 1/sqrt(n-3); required |z_r| = (z_alpha+z_beta)*SE
    z_required = (z_alpha + z_beta) / torch.sqrt(torch.clamp(n - 3.0, min=1.0))
    # Inverse Fisher transform: r = tanh(z)
    r_mag = torch.tanh(torch.abs(z_required))

    if alt == "less":
        # return magnitude; user can apply sign if desired
        out_t = r_mag
    else:
        out_t = r_mag

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
