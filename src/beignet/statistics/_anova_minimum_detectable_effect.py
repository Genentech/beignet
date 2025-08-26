import math

import torch
from torch import Tensor

from ._anova_power import anova_power


def anova_minimum_detectable_effect(
    sample_size: Tensor,
    k: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Minimum detectable Cohen's f for one-way ANOVA.

    Parameters
    ----------
    sample_size : Tensor
        Total sample size across groups.
    k : Tensor
        Number of groups.
    power : float, default=0.8
        Target power.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    Tensor
        Minimal Cohen's f achieving the requested power.

    When to Use
    -----------
    **Traditional Statistics:**

    - **Study planning:** Determining the smallest effect size that can be reliably detected with available sample
    - **Power estimation for grant proposals:** Calculating minimal detectable differences for funding applications
    - **Experimental design optimization:** Understanding trade-offs between sample size and effect sensitivity
    - **Multi-group comparisons:** When comparing means across 3+ independent groups
    - **Factor analysis:** Testing main effects in single-factor experimental designs
    - **Resource allocation:** Optimizing study design given fixed sample size constraints

    **Machine Learning Applications:**

    - **A/B/C+ testing:** Minimum detectable lift in conversion rates across multiple treatment groups
    - **Feature importance validation:** Smallest effect size detectable when comparing feature contributions
    - **Model performance comparison:** Minimum meaningful difference between 3+ model variants
    - **Hyperparameter sensitivity analysis:** Detecting parameter effects across multiple configurations
    - **Multi-arm bandit optimization:** Effect size thresholds for algorithmic arm selection
    - **Recommendation system testing:** Minimum detectable engagement differences across algorithm variants
    - **Ad targeting effectiveness:** Smallest CTR improvements detectable across audience segments
    - **User segmentation validation:** Effect sizes for behavioral differences between user groups
    - **Content optimization:** Minimum detectable engagement differences across content variants
    - **Algorithm fairness testing:** Detecting performance disparities across demographic groups
    - **Ensemble method comparison:** Minimum performance gaps between multiple ensemble strategies
    - **Cross-validation strategy evaluation:** Effect size detection across different validation approaches
    - **Feature engineering impact:** Minimum detectable improvements from feature transformations
    - **Model deployment strategies:** Detecting performance differences across deployment configurations
    - **Data preprocessing comparison:** Effect size detection across different preprocessing pipelines

    **Interpretation Guidelines:**

    - **Cohen's f = 0.1:** Small effect (1% of variance explained)
    - **Cohen's f = 0.25:** Medium effect (6.25% of variance explained)
    - **Cohen's f = 0.4:** Large effect (16% of variance explained)
    - **Values < 0.1:** Very small effects, may lack practical significance
    - **Consider practical significance:** Statistical detectability doesn't guarantee business impact
    - **Resource constraints:** Balance between sample size costs and effect size sensitivity
    """
    N0 = torch.as_tensor(sample_size)
    k0 = torch.as_tensor(k)
    scalar_out = N0.ndim == 0 and k0.ndim == 0
    N = torch.atleast_1d(N0)
    k = torch.atleast_1d(k0)
    dtype = (
        torch.float64
        if (N.dtype == torch.float64 or k.dtype == torch.float64)
        else torch.float32
    )
    N = torch.clamp(N.to(dtype), min=3.0)
    k = torch.clamp(k.to(dtype), min=2.0)

    # Initial guess using rough normal approximation
    sqrt2 = math.sqrt(2.0)
    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2
    # df1 = k-1; f ≈ (zα+zβ) * sqrt(df1/N)
    df1 = torch.clamp(k - 1.0, min=1.0)
    f0 = torch.clamp((z_alpha + z_beta) * torch.sqrt(df1 / N), min=1e-8)

    f_lo = torch.zeros_like(f0) + 1e-8
    f_hi = torch.clamp(2.0 * f0 + 1e-6, min=1e-6)

    # Ensure upper bound meets target power
    for _ in range(8):
        p_hi = anova_power(f_hi, N, k, alpha)
        need_expand = p_hi < power
        if not torch.any(need_expand):
            break
        f_hi = torch.where(need_expand, f_hi * 2.0, f_hi)
        f_hi = torch.clamp(f_hi, max=torch.tensor(10.0, dtype=dtype))

    f = (f_lo + f_hi) * 0.5
    for _ in range(24):
        p_mid = anova_power(f, N, k, alpha)
        go_right = p_mid < power
        f_lo = torch.where(go_right, f, f_lo)
        f_hi = torch.where(go_right, f_hi, f)
        f = (f_lo + f_hi) * 0.5

    out_t = torch.clamp(f, min=0.0)
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
