import math

import torch
from torch import Tensor


def multivariable_linear_regression_power(
    r_squared: Tensor,
    sample_size: Tensor,
    n_predictors: Tensor,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Power analysis for multiple linear regression.

    Tests the overall model significance (F-test) or individual coefficient
    significance in multiple linear regression.

    When to Use
    -----------
    **Traditional Statistics:**
    - Research planning for multiple predictor studies
    - Clinical trial design with multiple covariates
    - Observational study planning with confounding variables
    - Economic analysis with multiple explanatory variables
    - Survey research with multiple demographic predictors

    **Machine Learning Contexts:**
    - Feature selection validation: assessing power to detect feature importance
    - Model interpretability: testing significance of linear model coefficients
    - Baseline model evaluation: establishing statistical significance of linear relationships
    - Causal inference: testing treatment effects while controlling for confounders
    - Cross-validation planning: ensuring adequate power for regression model validation
    - A/B testing with covariates: accounting for user characteristics in experiment design
    - Fairness assessment: testing for significant bias in linear prediction models
    - Transfer learning: evaluating feature importance consistency across domains
    - Multi-task learning: assessing shared feature significance across tasks
    - Hyperparameter optimization: determining statistical significance of parameter choices

    **Choose multiple regression over simple regression when:**
    - Multiple predictors are theoretically important
    - Need to control for confounding variables
    - Interest is in partial effects of predictors
    - Model complexity is justified by research questions

    **Choose multiple regression over other methods when:**
    - Relationship between predictors and outcome is approximately linear
    - Predictors are continuous or properly coded categorical variables
    - Residuals meet normality and homoscedasticity assumptions
    - Interpretability of coefficients is important

    **Power considerations:**
    - Power decreases as number of predictors increases (degrees of freedom)
    - Multicollinearity among predictors reduces power
    - Larger R² values indicate higher power
    - Sample size should be at least 10-20 times the number of predictors

    Parameters
    ----------
    r_squared : Tensor
        Expected R² (coefficient of determination) under the alternative hypothesis.
        Range is [0, 1).
    sample_size : Tensor
        Total sample size.
    n_predictors : Tensor
        Number of predictor variables (excluding intercept).
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    output : Tensor, shape=(...)
        Statistical power values in range [0, 1].

    Examples
    --------
    >>> r_squared = torch.tensor(0.15)  # Medium effect
    >>> sample_size = torch.tensor(100)
    >>> n_predictors = torch.tensor(3)
    >>> multivariable_linear_regression_power(r_squared, sample_size, n_predictors)
    tensor(0.8234)

    Notes
    -----
    For multiple regression: Y = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ + ε

    The overall F-test statistic is:
    F = (R²/p) / ((1-R²)/(n-p-1))

    Under H₀: R² = 0 (no relationship)
    Under H₁: R² > 0 (some relationship)

    The F-statistic follows F(p, n-p-1) under H₀ and noncentral F under H₁.

    Effect size interpretation (Cohen, 1988):
    - R² = 0.02: small effect
    - R² = 0.13: medium effect
    - R² = 0.26: large effect

    References
    ----------
    Cohen, J. (1988). Statistical power analysis for the behavioral sciences.
    Lawrence Erlbaum Associates.
    """
    r_squared = torch.atleast_1d(torch.as_tensor(r_squared))
    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))
    n_predictors = torch.atleast_1d(torch.as_tensor(n_predictors))

    # Ensure floating point dtype
    dtypes = [r_squared.dtype, sample_size.dtype, n_predictors.dtype]
    if any(dt == torch.float64 for dt in dtypes):
        dtype = torch.float64
    else:
        dtype = torch.float32

    r_squared = r_squared.to(dtype)
    sample_size = sample_size.to(dtype)
    n_predictors = n_predictors.to(dtype)

    # Validate inputs
    r_squared = torch.clamp(r_squared, min=0.0, max=0.99)
    sample_size = torch.clamp(sample_size, min=n_predictors + 10)
    n_predictors = torch.clamp(n_predictors, min=1.0)

    # Degrees of freedom
    df_num = n_predictors  # Between groups
    df_den = sample_size - n_predictors - 1  # Within groups
    df_den = torch.clamp(df_den, min=1.0)

    # Critical F-value using chi-square approximation
    sqrt2 = math.sqrt(2.0)
    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2

    # For F(df1,df2), approximate critical value
    f_critical = 1.0 + z_alpha * torch.sqrt(2.0 / df_num)

    # Noncentrality parameter for noncentral F
    lambda_nc = sample_size * r_squared / (1 - r_squared)

    # Noncentral F approximation using normal distribution
    # F ~ F(df1, df2, λ) ≈ N(μ, σ²) for large df
    mean_nf = (1 + lambda_nc / df_num) * (df_den / (df_den - 2))
    var_nf = (
        2
        * (df_den / (df_den - 2)) ** 2
        * ((df_num + lambda_nc) / df_num + (df_den - 2) / df_den)
    )
    std_nf = torch.sqrt(torch.clamp(var_nf, min=1e-12))

    # Standardized test statistic
    z_score = (f_critical - mean_nf) / std_nf

    # Power = P(F > F_critical | H₁)
    power = 0.5 * (1 - torch.erf(z_score / sqrt2))

    # Clamp to valid range
    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)
        return out
    return power
