import math

import torch
from torch import Tensor


def repeated_measures_analysis_of_variance_power(
    effect_size: Tensor,
    n_subjects: Tensor,
    n_timepoints: Tensor,
    epsilon: Tensor = 1.0,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Power analysis for repeated measures ANOVA.

    Tests the within-subjects (time) effect in a repeated measures design
    using the F-test with sphericity correction.

    When to Use
    -----------
    **Traditional Statistics:**
    - Longitudinal studies with repeated measurements over time
    - Within-subject experimental designs with multiple conditions
    - Pre-post intervention studies with multiple time points
    - Clinical trials with repeated efficacy assessments
    - Learning studies measuring performance across sessions

    **Machine Learning Contexts:**
    - Model performance tracking: evaluating performance changes across training epochs
    - Online learning evaluation: assessing algorithm performance over time
    - Cross-validation with temporal dependency: accounting for time-series structure
    - Active learning: evaluating query strategy effectiveness across iterations
    - Hyperparameter optimization: comparing parameter settings across time periods
    - Ensemble method evaluation: tracking ensemble performance across updates
    - Domain adaptation: measuring model performance across temporal domains
    - Transfer learning: evaluating knowledge retention across sequential tasks
    - Federated learning: assessing model consistency across communication rounds
    - Time series forecasting: comparing model performance across forecast horizons

    **Choose repeated measures ANOVA over independent ANOVA when:**
    - Same subjects measured at multiple time points or conditions
    - Within-subject correlation is expected and should be accounted for
    - Higher power is needed (repeated measures increases power)
    - Individual differences should be controlled statistically

    **Choose repeated measures ANOVA over mixed-effects models when:**
    - Balanced design with no missing data
    - Simple repeated measures structure (no complex nesting)
    - Traditional ANOVA assumptions are met
    - Sphericity assumption is reasonable or correctable

    **Design considerations:**
    - Power increases with number of repeated measures
    - Power depends on within-subject correlation (higher correlation = higher power)
    - Sphericity violations reduce power and require corrections
    - Compound symmetry assumption may be violated

    Parameters
    ----------
    effect_size : Tensor
        Cohen's f for the within-subjects effect.
    n_subjects : Tensor
        Number of subjects.
    n_timepoints : Tensor
        Number of time points (repeated measurements per subject).
    epsilon : Tensor, default=1.0
        Sphericity correction factor (Greenhouse-Geisser or Huynh-Feldt).
        Range is [1/(k-1), 1] where k = n_timepoints.
        epsilon = 1.0 assumes perfect sphericity.
        epsilon < 1.0 indicates sphericity violation.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    output : Tensor, shape=(...)
        Statistical power values in range [0, 1].

    Examples
    --------
    >>> effect_size = torch.tensor(0.25)
    >>> n_subjects = torch.tensor(20)
    >>> n_timepoints = torch.tensor(4)
    >>> repeated_measures_analysis_of_variance_power(effect_size, n_subjects, n_timepoints)
    tensor(0.7891)

    Notes
    -----
    Repeated measures ANOVA model:
    Y_ij = μ + α_i + S_j + (αS)_ij + ε_ij

    Where:
    - Y_ij = observation for subject j at time i
    - μ = grand mean
    - α_i = time effect
    - S_j = subject effect (random)
    - (αS)_ij = time × subject interaction (error)
    - ε_ij = residual error

    The within-subjects F-test:
    F = MS_time / MS_error

    Degrees of freedom:
    - df_time = k - 1 (where k = n_timepoints)
    - df_error = (n - 1)(k - 1) (where n = n_subjects)

    With sphericity correction:
    - df_time × ε and df_error × ε

    Effect size Cohen's f interpretation:
    - f = 0.10: small effect
    - f = 0.25: medium effect
    - f = 0.40: large effect

    References
    ----------
    Maxwell, S. E., & Delaney, H. D. (2004). Designing experiments and
    analyzing data: a model comparison perspective. Lawrence Erlbaum Associates.
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    n_subjects = torch.atleast_1d(torch.as_tensor(n_subjects))
    n_timepoints = torch.atleast_1d(torch.as_tensor(n_timepoints))
    epsilon = torch.atleast_1d(torch.as_tensor(epsilon))

    dtypes = [effect_size.dtype, n_subjects.dtype, n_timepoints.dtype, epsilon.dtype]
    if any(dt == torch.float64 for dt in dtypes):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    n_subjects = n_subjects.to(dtype)
    n_timepoints = n_timepoints.to(dtype)
    epsilon = epsilon.to(dtype)

    effect_size = torch.clamp(effect_size, min=0.0)
    n_subjects = torch.clamp(n_subjects, min=3.0)
    n_timepoints = torch.clamp(n_timepoints, min=2.0)

    epsilon_min = 1.0 / (n_timepoints - 1.0)
    epsilon = torch.maximum(epsilon, epsilon_min)
    epsilon = torch.clamp(epsilon, max=1.0)

    df_time = n_timepoints - 1.0
    df_error = (n_subjects - 1.0) * (n_timepoints - 1.0)

    df_time_corrected = df_time * epsilon
    df_error_corrected = df_error * epsilon
    df_error_corrected = torch.clamp(df_error_corrected, min=1.0)

    lambda_nc = n_subjects * (effect_size**2) * n_timepoints

    sqrt2 = math.sqrt(2.0)
    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
    chi2_critical = df_time_corrected + z_alpha * torch.sqrt(2 * df_time_corrected)
    f_critical = chi2_critical / df_time_corrected

    mean_nc_chi2 = df_time_corrected + lambda_nc
    var_nc_chi2 = 2 * (df_time_corrected + 2 * lambda_nc)
    mean_f = mean_nc_chi2 / df_time_corrected
    var_f = var_nc_chi2 / (df_time_corrected**2)

    var_f = var_f * (
        (df_error_corrected + 2.0) / torch.clamp(df_error_corrected, min=1.0)
    )

    std_f = torch.sqrt(torch.clamp(var_f, min=1e-12))

    z_score = (f_critical - mean_f) / std_f

    power = 0.5 * (1 - torch.erf(z_score / sqrt2))

    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)
        return out
    return power
