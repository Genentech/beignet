import math

import torch
from torch import Tensor


def analysis_of_covariance_power(
    effect_size: Tensor,
    sample_size: Tensor,
    k: Tensor,
    covariate_r2: Tensor,
    n_covariates: Tensor | int = 1,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Power for fixed-effects ANCOVA (one-way) with covariate adjustment.

    This function computes statistical power for Analysis of Covariance designs
    where treatment effects are tested after accounting for one or more covariates.
    ANCOVA increases power by reducing error variance through covariate adjustment.

    This function is differentiable with respect to all tensor parameters.
    While traditional power analysis doesn't require gradients, differentiability
    enables integration into machine learning pipelines where effect sizes or
    covariate relationships might be learned parameters.

    When to Use
    -----------
    **Traditional Statistics:**
    - Clinical trials with baseline measurements as covariates
    - Educational research controlling for pre-test scores
    - Agricultural experiments controlling for soil quality or plot characteristics
    - Psychological studies controlling for individual difference variables
    - Medical research controlling for age, gender, or other patient characteristics
    - Quality control studies controlling for batch or environmental factors

    **Machine Learning Contexts:**
    - Model comparison controlling for dataset characteristics or complexity
    - A/B testing with user characteristics as covariates
    - Hyperparameter optimization controlling for computational resources
    - Cross-validation controlling for data distribution characteristics
    - Transfer learning evaluation controlling for source domain features
    - Ensemble method comparison controlling for base model characteristics
    - Active learning evaluation controlling for initial training set properties
    - Domain adaptation studies controlling for domain similarity measures
    - Fairness evaluation controlling for protected attribute correlations
    - Causal inference with observational confounders as covariates
    - Recommendation system evaluation controlling for user behavior patterns
    - Computer vision model comparison controlling for image complexity metrics

    **Choose ANCOVA over ANOVA when:**
    - Have continuous covariates that correlate with the outcome
    - Want to increase statistical power by reducing error variance
    - Need to control for confounding variables
    - Baseline measurements are available before treatment assignment
    - Individual differences need to be accounted for in group comparisons

    **Choose ANCOVA over regression when:**
    - Primary interest is in group differences (categorical predictors)
    - Treatment groups are the main focus, not individual covariate effects
    - Need to test overall group effect after covariate adjustment
    - Want to increase power for group comparisons specifically

    **Interpretation Guidelines:**
    - Power increases as covariate R² increases (better error variance reduction)
    - Effect size should be Cohen's f for the group effect before covariate adjustment
    - Higher covariate R² means smaller sample sizes needed for same power
    - Number of covariates affects degrees of freedom and critical values
    - R² represents proportion of outcome variance explained by covariates
    - Assumes linear relationship between covariates and outcome
    - Assumes homogeneity of regression slopes across groups

    Parameters
    ----------
    effect_size : Tensor
        Cohen's f for the group effect (as in ANOVA, before covariate adjustment).
    sample_size : Tensor
        Total sample size N across groups.
    k : Tensor
        Number of groups.
    covariate_r2 : Tensor
        Proportion of outcome variance explained by covariates (R² in [0,1)).
    n_covariates : Tensor or int, default=1
        Number of covariates included in the model.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    Tensor
        Statistical power.

    Notes
    -----
    Uses a noncentral F approximation similar to ANOVA, with the noncentrality
    parameter inflated by the error variance reduction: λ ≈ N * f² / (1 - R²).
    Degrees of freedom: df1 = k - 1, df2 = N - k - p, where p = n_covariates.
    """
    f = torch.atleast_1d(torch.as_tensor(effect_size))
    N = torch.atleast_1d(torch.as_tensor(sample_size))
    k = torch.atleast_1d(torch.as_tensor(k))
    R2 = torch.atleast_1d(torch.as_tensor(covariate_r2))
    p = torch.atleast_1d(torch.as_tensor(n_covariates))

    # dtype management
    dtype = (
        torch.float64
        if any(t.dtype == torch.float64 for t in (f, N, k, R2, p))
        else torch.float32
    )
    f = torch.clamp(f.to(dtype), min=0.0)
    N = torch.clamp(N.to(dtype), min=3.0)
    k = torch.clamp(k.to(dtype), min=2.0)
    R2 = torch.clamp(R2.to(dtype), min=0.0, max=1 - torch.finfo(dtype).eps)
    p = torch.clamp(p.to(dtype), min=0.0)

    # Degrees of freedom
    df1 = torch.clamp(k - 1.0, min=1.0)
    df2 = torch.clamp(N - k - p, min=1.0)

    # Critical F via chi-square normal approximation (consistent with ANOVA impl)
    sqrt2 = math.sqrt(2.0)
    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
    chi2_crit = df1 + z_alpha * torch.sqrt(2 * df1)
    f_crit = chi2_crit / df1

    # Noncentrality parameter with variance reduction
    lambda_nc = N * f**2 / torch.clamp(1.0 - R2, min=torch.finfo(dtype).eps)

    # Approximate noncentral F via noncentral chi-square moments
    mean_nc_chi2 = df1 + lambda_nc
    var_nc_chi2 = 2 * (df1 + 2 * lambda_nc)
    mean_f = mean_nc_chi2 / df1
    var_f = var_nc_chi2 / (df1**2)

    # Finite df2 adjustment (smooth)
    var_f = var_f * ((df2 + 2.0) / torch.clamp(df2, min=1.0))

    std_f = torch.sqrt(var_f)
    z = (f_crit - mean_f) / torch.clamp(std_f, min=1e-10)
    power = 0.5 * (1 - torch.erf(z / sqrt2))

    out_t = torch.clamp(power, 0.0, 1.0)
    if out is not None:
        out.copy_(out_t)
        return out
    return out_t
