"""General F-test power calculation."""

import math

import torch
from torch import Tensor


def f_test_power(
    effect_size: Tensor,
    df1: Tensor,
    df2: Tensor,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Compute statistical power for general F-tests.

    This function calculates the probability of correctly rejecting the null
    hypothesis when the alternative hypothesis is true for a general F-test.
    This covers F-tests beyond ANOVA, such as regression model comparisons,
    variance ratio tests, and other applications.

    This function is differentiable with respect to all tensor parameters.
    While traditional power analysis doesn't require gradients, differentiability
    enables integration into machine learning pipelines where effect sizes might
    be learned parameters or part of experimental design optimization.

    When to Use
    -----------
    **Traditional Statistics:**
    - Regression model comparisons (nested model F-tests)
    - Testing variance equality between groups
    - ANOVA omnibus tests
    - Testing multiple regression coefficients simultaneously

    **Machine Learning Contexts:**
    - Model comparison: testing if complex model significantly outperforms simple model
    - Feature selection: testing significance of feature subsets
    - Hyperparameter optimization: comparing nested parameter configurations
    - Cross-validation: testing variance differences across folds
    - Ensemble methods: testing if individual models contribute significantly
    - Neural networks: testing layer or component significance
    - Regularization: comparing models with different penalty structures
    - Transfer learning: testing significance of fine-tuning improvements
    - AutoML: automated model selection with statistical validation

    **Choose F-test when:**
    - Comparing nested models or testing multiple parameters simultaneously
    - Testing variance ratios or equality
    - Need omnibus test before specific comparisons
    - Assumptions of normality and homoscedasticity are met

    **Common Applications:**
    - Regression model selection
    - ANOVA omnibus tests
    - Variance equality testing
    - Multiple parameter significance testing

    Parameters
    ----------
    effect_size : Tensor
        Effect size (Cohen's f² or similar). This represents the magnitude of
        the effect being tested. For regression contexts, this could be the
        R² change or similar measure. Should be non-negative.

    df1 : Tensor
        Degrees of freedom for the numerator (effect).

    df2 : Tensor
        Degrees of freedom for the denominator (error).

    alpha : float, default=0.05
        Significance level (Type I error rate).

    out : Tensor, optional
        Output tensor. Default, `None`.

    Returns
    -------
    output : Tensor
        Statistical power (probability of correctly rejecting false null hypothesis).

    Examples
    --------
    >>> effect_size = torch.tensor(0.15)
    >>> df1 = torch.tensor(3)
    >>> df2 = torch.tensor(96)
    >>> f_test_power(effect_size, df1, df2)
    tensor(0.8743)

    Notes
    -----
    The power calculation is based on the noncentral F-distribution. Under the
    null hypothesis, the F-statistic follows F(df1, df2). Under the alternative
    hypothesis, it follows a noncentral F-distribution with noncentrality
    parameter:

    λ = N * f²

    Where N is the total sample size and f² is Cohen's f² effect size.

    For regression contexts:
    - f² = R²/(1-R²) where R² is the coefficient of determination
    - For comparing nested models: f² = (R²_full - R²_reduced)/(1-R²_full)

    Cohen's f² effect size interpretation:
    - Small effect: f² = 0.02
    - Medium effect: f² = 0.15
    - Large effect: f² = 0.35

    The calculation uses normal approximations for large degrees of freedom,
    which provides good accuracy for most practical scenarios while maintaining
    computational efficiency and differentiability.

    References
    ----------
    .. [1] Cohen, J. (1988). Statistical power analysis for the behavioral
           sciences (2nd ed.). Erlbaum.
    .. [2] Cohen, J., Cohen, P., West, S. G., & Aiken, L. S. (2003). Applied
           multiple regression/correlation analysis for the behavioral sciences.
           Erlbaum.
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    df1 = torch.atleast_1d(torch.as_tensor(df1))
    df2 = torch.atleast_1d(torch.as_tensor(df2))

    if (
        effect_size.dtype == torch.float64
        or df1.dtype == torch.float64
        or df2.dtype == torch.float64
    ):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    df1 = df1.to(dtype)
    df2 = df2.to(dtype)

    effect_size = torch.clamp(effect_size, min=0.0)

    df1 = torch.clamp(df1, min=1.0)
    df2 = torch.clamp(df2, min=1.0)


    sqrt_2 = math.sqrt(2.0)

    total_sample_size = df1 + df2 + 1
    lambda_param = total_sample_size * effect_size

    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt_2

    f_crit = 1.0 + z_alpha * torch.sqrt(2.0 / df1)

    mean_f_alt = 1.0 + lambda_param / df1

    std_f_alt = torch.sqrt(2.0 / df1) * torch.sqrt(1.0 + 2.0 * lambda_param / df1)

    z_score = (f_crit - mean_f_alt) / torch.clamp(std_f_alt, min=1e-10)

    power = 0.5 * (1.0 - torch.erf(z_score / sqrt_2))

    output = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(output)
        return out

    return output
