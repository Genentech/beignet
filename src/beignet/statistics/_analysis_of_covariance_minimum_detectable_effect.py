import math

import torch
from torch import Tensor

from ._analysis_of_covariance_power import analysis_of_covariance_power


def analysis_of_covariance_minimum_detectable_effect(
    sample_size: Tensor,
    groups: Tensor,
    covariate_r2: Tensor,
    n_covariates: Tensor | int = 1,
    power: float = 0.8,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Minimum detectable Cohen's f for fixed-effects ANCOVA (one-way).

    Computes the smallest effect size that can be reliably detected with
    specified sample size, power, and significance level for Analysis of
    Covariance designs with covariate adjustment.

    This function is differentiable with respect to all tensor parameters.
    While traditional effect size calculations don't require gradients,
    differentiability enables integration into machine learning pipelines where
    sample sizes or covariate relationships might be learned parameters.

    When to Use
    -----------
    **Traditional Statistics:**
    - Post-hoc analysis: determining what effect size was detectable in completed studies
    - Study planning: understanding detection limits for available sample size
    - Grant applications: justifying feasibility of detecting meaningful effects
    - Meta-analysis: understanding heterogeneity in detectable effect sizes across studies
    - Clinical trial interpretation: determining if null results indicate no effect or insufficient power
    - Quality assessment: evaluating whether studies had adequate power for meaningful effects

    **Machine Learning Contexts:**
    - Model comparison feasibility: determining minimum performance differences detectable
    - A/B testing design: understanding detection limits for user behavior changes
    - Hyperparameter optimization: determining minimum parameter effect sizes detectable
    - Cross-validation interpretation: understanding what performance differences are meaningful
    - Transfer learning evaluation: determining minimum adaptation effects detectable
    - Ensemble method analysis: understanding minimum component contribution detectable
    - Active learning assessment: determining minimum query strategy improvements detectable
    - Domain adaptation studies: understanding minimum domain effects detectable
    - Fairness evaluation: determining minimum bias effects detectable across groups
    - Causal inference: understanding minimum treatment effects detectable with confounders
    - Feature importance analysis: determining minimum feature contributions detectable

    **Choose MDE over power analysis when:**
    - Sample size and study design are already fixed
    - Want to understand detection capabilities of existing data
    - Need to interpret negative results or small effects
    - Evaluating feasibility of detecting theoretically important effect sizes
    - Comparing detection capabilities across different study designs

    **Choose ANCOVA MDE over ANOVA MDE when:**
    - Have relevant covariates that explain outcome variance
    - Want to understand power gains from covariate adjustment
    - Need to account for confounding variables in effect size interpretation
    - Baseline measurements or individual differences are available

    **Interpretation Guidelines:**
    - Smaller MDE indicates better detection capability
    - Higher covariate R² reduces MDE (increases detection power)
    - Effects smaller than MDE are unlikely to be detected reliably
    - MDE represents Cohen's f: 0.10 (small), 0.25 (medium), 0.40 (large)
    - Consider practical significance: is MDE smaller than meaningful effect?
    - MDE decreases as sample size increases and covariate correlation increases

    Parameters
    ----------
    sample_size : Tensor
        Total N across groups.
    groups : Tensor
        Number of groups.
    covariate_r2 : Tensor
        R² of covariates (variance explained), in [0, 1).
    n_covariates : Tensor or int, default=1
        Number of covariates.
    power : float, default=0.8
        Target power.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    Tensor
        Minimal Cohen's f achieving the requested power.
    """
    N0 = torch.as_tensor(sample_size)
    groups0 = torch.as_tensor(groups)
    R20 = torch.as_tensor(covariate_r2)
    num_covariates0 = torch.as_tensor(n_covariates)
    scalar_out = (
        N0.ndim == 0
        and groups0.ndim == 0
        and R20.ndim == 0
        and num_covariates0.ndim == 0
    )
    N = torch.atleast_1d(N0)
    groups = torch.atleast_1d(groups0)
    R2 = torch.atleast_1d(R20)
    num_covariates = torch.atleast_1d(num_covariates0)

    dtype = (
        torch.float64
        if any(t.dtype == torch.float64 for t in (N, groups, R2, num_covariates))
        else torch.float32
    )
    N = torch.clamp(N.to(dtype), min=3.0)
    groups = torch.clamp(groups.to(dtype), min=2.0)
    R2 = torch.clamp(R2.to(dtype), min=0.0, max=1 - torch.finfo(dtype).eps)
    num_covariates = torch.clamp(num_covariates.to(dtype), min=0.0)

    # Initial guess using df1 = groups-1 and variance reduction 1-R2
    sqrt2 = math.sqrt(2.0)
    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2
    df1 = torch.clamp(groups - 1.0, min=1.0)
    # Effective N for noncentrality: λ = N f^2/(1-R2)
    effect_size_f0 = torch.clamp(
        (z_alpha + z_beta)
        * torch.sqrt(df1 / torch.clamp(N, min=1.0))
        * torch.sqrt(torch.clamp(1.0 - R2, min=torch.finfo(dtype).eps)),
        min=1e-8,
    )

    effect_size_f_lo = torch.zeros_like(effect_size_f0) + 1e-8
    effect_size_f_hi = torch.clamp(2.0 * effect_size_f0 + 1e-6, min=1e-6)

    for _ in range(8):
        p_hi = analysis_of_covariance_power(
            effect_size_f_hi, N, groups, R2, num_covariates, alpha
        )
        need_expand = p_hi < power
        if not torch.any(need_expand):
            break
        effect_size_f_hi = torch.where(
            need_expand, effect_size_f_hi * 2.0, effect_size_f_hi
        )
        effect_size_f_hi = torch.clamp(
            effect_size_f_hi, max=torch.tensor(10.0, dtype=dtype)
        )

    effect_size_f = (effect_size_f_lo + effect_size_f_hi) * 0.5
    for _ in range(24):
        p_mid = analysis_of_covariance_power(
            effect_size_f, N, groups, R2, num_covariates, alpha
        )
        go_right = p_mid < power
        effect_size_f_lo = torch.where(go_right, effect_size_f, effect_size_f_lo)
        effect_size_f_hi = torch.where(go_right, effect_size_f_hi, effect_size_f)
        effect_size_f = (effect_size_f_lo + effect_size_f_hi) * 0.5

    out_t = torch.clamp(effect_size_f, min=0.0)
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
