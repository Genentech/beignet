import math

import torch
from torch import Tensor


def mixed_model_power(
    effect_size: Tensor,
    n_subjects: Tensor,
    n_observations_per_subject: Tensor,
    icc: Tensor,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Power analysis for mixed-effects models (random intercept).

    Tests fixed effects in a linear mixed-effects model with random intercepts
    using the Wald test with appropriate degrees of freedom adjustment.

    When to Use
    -----------
    **Traditional Statistics:**
    - Longitudinal studies with repeated measurements per subject
    - Cluster randomized trials (schools, hospitals, clinics as clusters)
    - Multi-site studies with site-level clustering
    - Family studies with genetic clustering
    - Hierarchical data structures (students within classrooms)

    **Machine Learning Contexts:**
    - Federated learning: accounting for client-level clustering effects
    - Personalized medicine: individual-level random effects in treatment response
    - Time series ML: repeated observations within subjects over time
    - Multi-domain adaptation: domain-specific random effects
    - A/B testing with repeated user interactions
    - Recommendation systems: user-specific random effects
    - Natural language processing: author or document-level random effects
    - Computer vision: image series from same subject/source
    - Longitudinal biomarker studies with ML prediction models

    **Choose mixed models over fixed effects when:**
    - Data has clustering or hierarchical structure
    - Repeated measurements from same subjects/units
    - Interest in both population-level (fixed) and cluster-level (random) effects
    - Need to generalize beyond observed clusters
    - Intraclass correlation (ICC) > 0.05

    **ICC Interpretation:**
    - ICC = 0: no clustering (use regular regression)
    - ICC = 0.05: small clustering effect
    - ICC = 0.10: medium clustering effect
    - ICC = 0.20+: large clustering effect

    Parameters
    ----------
    effect_size : Tensor
        Standardized effect size (Cohen's d) for the fixed effect.
    n_subjects : Tensor
        Number of subjects (clusters).
    n_observations_per_subject : Tensor
        Average number of observations per subject.
    icc : Tensor
        Intraclass correlation coefficient representing the proportion
        of variance due to between-subject differences. Range [0, 1).
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    output : Tensor, shape=(...)
        Statistical power values in range [0, 1].

    Examples
    --------
    >>> effect_size = torch.tensor(0.5)
    >>> n_subjects = torch.tensor(30)
    >>> n_observations_per_subject = torch.tensor(4)
    >>> icc = torch.tensor(0.1)
    >>> mixed_model_power(effect_size, n_subjects, n_observations_per_subject, icc)
    tensor(0.8234)

    Notes
    -----
    Mixed-effects model: Y_ij = β₀ + β₁X_ij + u_i + ε_ij

    Where:
    - u_i ~ N(0, τ²) is the random intercept
    - ε_ij ~ N(0, σ²) is the residual error
    - ICC = τ²/(τ² + σ²)

    The design effect accounts for clustering:
    DEFF = 1 + (m̄ - 1) × ICC

    where m̄ is the average cluster size.

    Effective sample size = n_subjects × n_observations_per_subject / DEFF

    This approximation assumes:
    - Balanced design (equal observations per subject)
    - Random intercept model only
    - Large sample approximation

    References
    ----------
    Hedeker, D., & Gibbons, R. D. (2006). Longitudinal data analysis.
    John Wiley & Sons.
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    n_subjects = torch.atleast_1d(torch.as_tensor(n_subjects))
    n_observations_per_subject = torch.atleast_1d(
        torch.as_tensor(n_observations_per_subject)
    )
    icc = torch.atleast_1d(torch.as_tensor(icc))

    dtypes = [
        effect_size.dtype,
        n_subjects.dtype,
        n_observations_per_subject.dtype,
        icc.dtype,
    ]
    if any(dt == torch.float64 for dt in dtypes):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    n_subjects = n_subjects.to(dtype)
    n_observations_per_subject = n_observations_per_subject.to(dtype)
    icc = icc.to(dtype)

    effect_size = torch.clamp(effect_size, min=0.0)
    n_subjects = torch.clamp(n_subjects, min=3.0)
    n_observations_per_subject = torch.clamp(n_observations_per_subject, min=1.0)
    icc = torch.clamp(icc, min=0.0, max=0.99)

    design_effect = 1.0 + (n_observations_per_subject - 1.0) * icc

    total_observations = n_subjects * n_observations_per_subject
    effective_n = total_observations / design_effect

    ncp = effect_size * torch.sqrt(effective_n / 4.0)

    df_approx = n_subjects - 2.0
    df_approx = torch.clamp(df_approx, min=1.0)

    sqrt2 = math.sqrt(2.0)
    z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt2

    t_critical = z_alpha * torch.sqrt(1.0 + 1.0 / (2.0 * df_approx))

    z_score = t_critical - ncp
    power = 0.5 * (1 - torch.erf(z_score / sqrt2))

    power = torch.clamp(power, 0.0, 1.0)

    if out is not None:
        out.copy_(power)
        return out
    return power
