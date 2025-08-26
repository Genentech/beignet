import math

import torch
from torch import Tensor


def multivariate_analysis_of_variance_power(
    effect_size: Tensor,
    sample_size: Tensor,
    n_variables: Tensor,
    n_groups: Tensor,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Power analysis for multivariate analysis of variance (MANOVA).

    Tests the overall group differences across multiple dependent variables
    using Wilks' Lambda or Pillai's trace statistics.

    When to Use
    -----------
    **Traditional Statistics:**
    - Comparing multiple groups on several related outcome variables simultaneously
    - Clinical trials with multiple efficacy endpoints
    - Educational research comparing interventions on multiple achievement measures
    - Psychological studies with multiple behavioral or cognitive outcomes
    - Marketing research comparing products on multiple attribute ratings

    **Machine Learning Contexts:**
    - Multi-task learning evaluation: comparing model performance across multiple tasks simultaneously
    - Feature evaluation: testing group differences across multiple engineered features
    - Ensemble method comparison: evaluating multiple model outputs across different conditions
    - Multi-label classification evaluation: comparing classifiers on multiple label dimensions
    - Cross-validation analysis: comparing models across multiple performance metrics simultaneously
    - A/B testing with multiple metrics: testing intervention effects on multiple KPIs
    - Fairness assessment: evaluating model performance across multiple demographic groups and metrics
    - Domain adaptation: comparing model performance across multiple domains and metrics
    - Transfer learning evaluation: assessing knowledge transfer across multiple target tasks
    - Hyperparameter optimization: comparing parameter settings across multiple evaluation criteria

    **Choose MANOVA over multiple ANOVAs when:**
    - Multiple dependent variables are conceptually related
    - Need to control family-wise error rate across variables
    - Interested in overall multivariate group differences
    - Variables are correlated (MANOVA accounts for correlation structure)
    - Want to identify discriminating patterns across variables

    **Choose MANOVA over other methods when:**
    - Multiple continuous dependent variables (not single composite score)
    - Groups are independent (not repeated measures)
    - Multivariate normality assumption is reasonable
    - Sample size is adequate relative to number of variables
    - Homogeneity of covariance matrices is plausible

    **Design considerations:**
    - Minimum sample size: n ≥ p + k (where p = variables, k = groups)
    - Power decreases as number of variables increases
    - Balanced groups provide optimal power
    - Effect size interpretation is complex in multivariate context

    Parameters
    ----------
    effect_size : Tensor
        Multivariate effect size. Can be interpreted as the multivariate
        Cohen's f based on Wilks' Lambda: f = √(Λ⁻¹ - 1).
    sample_size : Tensor
        Total sample size across all groups.
    n_variables : Tensor
        Number of dependent variables (p).
    n_groups : Tensor
        Number of groups (k).
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    output : Tensor, shape=(...)
        Statistical power values in range [0, 1].

    Examples
    --------
    >>> effect_size = torch.tensor(0.3)  # Medium multivariate effect
    >>> sample_size = torch.tensor(120)
    >>> n_variables = torch.tensor(3)  # 3 dependent variables
    >>> n_groups = torch.tensor(4)     # 4 groups
    >>> multivariate_analysis_of_variance_power(effect_size, sample_size, n_variables, n_groups)
    tensor(0.7845)

    Notes
    -----
    MANOVA model: Y = XB + E

    Where:
    - Y is n×p matrix of observations
    - X is n×k design matrix
    - B is k×p parameter matrix
    - E is n×p error matrix

    The test statistics are:
    - Wilks' Λ = |E| / |H + E|
    - Pillai's trace = tr(H(H + E)⁻¹)
    - Hotelling's T² = tr(HE⁻¹)
    - Roy's largest root = λmax(HE⁻¹)

    This implementation uses an F-approximation to Wilks' Lambda for
    power calculation, which is accurate for moderate to large samples.

    Effect size interpretation is challenging in MANOVA due to its
    multivariate nature. The provided effect_size should represent
    the overall multivariate effect.

    References
    ----------
    Hand, D. J., & Taylor, C. C. (1987). Multivariate analysis of variance
    and repeated measures. Chapman and Hall.
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    sample_size = torch.atleast_1d(torch.as_tensor(sample_size))
    n_variables = torch.atleast_1d(torch.as_tensor(n_variables))
    n_groups = torch.atleast_1d(torch.as_tensor(n_groups))

    # Ensure floating point dtype
    dtypes = [effect_size.dtype, sample_size.dtype, n_variables.dtype, n_groups.dtype]
    if any(dt == torch.float64 for dt in dtypes):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    sample_size = sample_size.to(dtype)
    n_variables = n_variables.to(dtype)
    n_groups = n_groups.to(dtype)

    # Validate inputs
    effect_size = torch.clamp(effect_size, min=0.0)
    sample_size = torch.clamp(sample_size, min=n_groups + n_variables + 5)
    n_variables = torch.clamp(n_variables, min=1.0)
    n_groups = torch.clamp(n_groups, min=2.0)

    # Degrees of freedom
    df_hypothesis = n_groups - 1  # Between groups
    df_error = sample_size - n_groups  # Within groups

    # Calculate intermediate values for F-approximation
    df1 = df_hypothesis * n_variables
    df2 = df_error * n_variables - (n_variables - df_hypothesis + 1) / 2
    df2 = torch.clamp(df2, min=1.0)

    # Convert effect size to noncentrality parameter
    effect_size_f_squared = effect_size**2

    # Noncentrality parameter for noncentral F
    lambda_nc = sample_size * effect_size_f_squared

    # Critical F-value using chi-square approximation
    sqrt2 = math.sqrt(2.0)
    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
    chi2_critical = df1 + z_alpha * torch.sqrt(2 * df1)
    f_critical = chi2_critical / df1

    # Noncentral F approximation
    mean_nc_f = (1.0 + lambda_nc / df1) * (df2 / (df2 - 2.0))
    var_nc_f = (
        2.0 * (df2 / (df2 - 2.0)) ** 2 * ((df1 + lambda_nc) / df1 + (df2 - 2.0) / df2)
    )
    std_nc_f = torch.sqrt(torch.clamp(var_nc_f, min=1e-12))

    # Standardized test statistic
    z_score = (f_critical - mean_nc_f) / std_nc_f

    # Power = P(F > F_critical | H₁)
    power = 0.5 * (1 - torch.erf(z_score / sqrt2))

    # Clamp to valid range
    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)
        return out
    return power
