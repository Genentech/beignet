import math

import torch
from torch import Tensor

from ._repeated_measures_analysis_of_variance_power import (
    repeated_measures_analysis_of_variance_power,
)


def repeated_measures_analysis_of_variance_sample_size(
    effect_size: Tensor,
    n_timepoints: Tensor,
    epsilon: Tensor = 1.0,
    power: float = 0.8,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Required number of subjects for repeated measures ANOVA.

    Calculates the number of subjects needed to achieve desired power
    for detecting within-subjects effects in repeated measures designs.

    Parameters
    ----------
    effect_size : Tensor
        Cohen's f for the within-subjects effect.
    n_timepoints : Tensor
        Number of time points (repeated measurements per subject).
    epsilon : Tensor, default=1.0
        Sphericity correction factor (Greenhouse-Geisser or Huynh-Feldt).
        Range is [1/(k-1), 1] where k = n_timepoints.
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
    >>> effect_size = torch.tensor(0.25)  # Medium effect
    >>> n_timepoints = torch.tensor(4)    # 4 time points
    >>> repeated_measures_analysis_of_variance_sample_size(effect_size, n_timepoints)
    tensor(18.0)

    Notes
    -----
    This function uses an iterative approach to find the number of subjects
    that achieves the desired power for the repeated measures ANOVA.

    Advantages of repeated measures designs:
    - Increased power due to reduced error variance
    - Fewer subjects needed compared to between-subjects designs
    - Control for individual differences

    Assumptions:
    - Sphericity (or appropriate correction applied)
    - Normality of residuals
    - No missing data patterns that bias results
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    n_timepoints = torch.atleast_1d(torch.as_tensor(n_timepoints))
    epsilon = torch.atleast_1d(torch.as_tensor(epsilon))

    # Ensure floating point dtype
    dtypes = [effect_size.dtype, n_timepoints.dtype, epsilon.dtype]
    if any(dt == torch.float64 for dt in dtypes):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    n_timepoints = n_timepoints.to(dtype)
    epsilon = epsilon.to(dtype)

    # Validate inputs
    effect_size = torch.clamp(effect_size, min=1e-8)
    n_timepoints = torch.clamp(n_timepoints, min=2.0)

    # Sphericity correction bounds
    epsilon_min = 1.0 / (n_timepoints - 1.0)
    epsilon = torch.maximum(epsilon, epsilon_min)
    epsilon = torch.clamp(epsilon, max=1.0)

    # Initial approximation
    sqrt2 = math.sqrt(2.0)
    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2

    # Initial guess based on F-test power
    # For repeated measures, the efficiency gain is approximately k-fold
    # where k is the number of time points
    efficiency_factor = n_timepoints * epsilon

    # Initial sample size approximation
    n_init = ((z_alpha + z_beta) / effect_size) ** 2 / efficiency_factor
    n_init = torch.clamp(n_init, min=5.0)

    # Iterative refinement
    n_current = n_init
    for _ in range(12):
        # Calculate current power
        current_power = repeated_measures_analysis_of_variance_power(
            effect_size, n_current, n_timepoints, epsilon, alpha=alpha
        )

        # Adjust sample size based on power gap
        power_gap = torch.clamp(power - current_power, min=-0.4, max=0.4)
        adjustment = 1.0 + 1.2 * power_gap
        n_current = torch.clamp(n_current * adjustment, min=5.0, max=1e5)

    # Round up to nearest integer
    n_out = torch.ceil(n_current)

    if out is not None:
        out.copy_(n_out)
        return out
    return n_out
