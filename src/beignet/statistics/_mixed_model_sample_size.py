import math

import torch
from torch import Tensor

from ._mixed_model_power import mixed_model_power


def mixed_model_sample_size(
    effect_size: Tensor,
    n_observations_per_subject: Tensor,
    icc: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Required number of subjects for mixed-effects models.

    Calculates the number of subjects (clusters) needed to achieve desired power
    for testing fixed effects in a linear mixed-effects model.

    Parameters
    ----------
    effect_size : Tensor
        Standardized effect size (Cohen's d) for the fixed effect.
    n_observations_per_subject : Tensor
        Average number of observations per subject.
    icc : Tensor
        Intraclass correlation coefficient. Range [0, 1).
    power : float, default=0.8
        Desired statistical power.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    output : Tensor
        Required number of subjects (rounded up to nearest integer).

    Examples
    --------
    >>> effect_size = torch.tensor(0.5)
    >>> n_observations_per_subject = torch.tensor(4)
    >>> icc = torch.tensor(0.1)
    >>> mixed_model_sample_size(effect_size, n_observations_per_subject, icc)
    tensor(28.0)

    Notes
    -----
    This function uses an iterative approach to find the number of subjects
    that achieves the desired power for the mixed-effects model.

    Sample size considerations:
    - Higher ICC requires more subjects (less efficient)
    - More observations per subject can partially compensate for clustering
    - Design effect = 1 + (m-1) × ICC reduces effective sample size
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    n_observations_per_subject = torch.atleast_1d(
        torch.as_tensor(n_observations_per_subject)
    )
    icc = torch.atleast_1d(torch.as_tensor(icc))

    dtypes = [effect_size.dtype, n_observations_per_subject.dtype, icc.dtype]
    if any(dt == torch.float64 for dt in dtypes):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    n_observations_per_subject = n_observations_per_subject.to(dtype)
    icc = icc.to(dtype)

    effect_size = torch.clamp(effect_size, min=1e-8)
    n_observations_per_subject = torch.clamp(n_observations_per_subject, min=1.0)
    icc = torch.clamp(icc, min=0.0, max=0.99)

    sqrt2 = math.sqrt(2.0)
    z_alpha = torch.erfinv(torch.tensor(1 - alpha / 2, dtype=dtype)) * sqrt2
    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2

    design_effect = 1.0 + (n_observations_per_subject - 1.0) * icc

    n_eff_needed = 4 * ((z_alpha + z_beta) / effect_size) ** 2

    n_subjects_init = n_eff_needed * design_effect / n_observations_per_subject
    n_subjects_init = torch.clamp(n_subjects_init, min=10.0)

    n_current = n_subjects_init
    for _ in range(15):
        current_power = mixed_model_power(
            effect_size, n_current, n_observations_per_subject, icc, alpha=alpha
        )

        power_gap = torch.clamp(power - current_power, min=-0.4, max=0.4)
        adjustment = 1.0 + 1.3 * power_gap
        n_current = torch.clamp(n_current * adjustment, min=10.0, max=1e5)

    n_out = torch.ceil(n_current)

    if out is not None:
        out.copy_(n_out)
        return out
    return n_out
