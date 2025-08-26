import math

import torch
from torch import Tensor

from ._friedman_test_power import friedman_test_power


def friedman_test_sample_size(
    effect_size: Tensor,
    n_treatments: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Required number of subjects for Friedman test.

    Calculates the number of subjects (blocks) needed to achieve desired power
    for detecting a specified effect size in a Friedman test.

    Parameters
    ----------
    effect_size : Tensor
        Effect size (variance of treatment effects relative to error variance).
    n_treatments : Tensor
        Number of treatments (repeated conditions) being compared.
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
    >>> effect_size = torch.tensor(0.4)
    >>> n_treatments = torch.tensor(3)
    >>> friedman_test_sample_size(effect_size, n_treatments)
    tensor(20.0)

    Notes
    -----
    This function uses an iterative approach to find the number of subjects
    that achieves the desired power. The calculation is based on the chi-square
    approximation to the Friedman test statistic.

    The Friedman test is particularly useful for:
    - Repeated measures designs with non-normal data
    - Small sample sizes where normality assumptions are questionable
    - Ordinal outcome measures
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    n_treatments = torch.atleast_1d(torch.as_tensor(n_treatments))

    # Ensure floating point dtype
    dtype = torch.promote_type(effect_size.dtype, n_treatments.dtype)
    if not dtype.is_floating_point:
        dtype = torch.float32
    effect_size = effect_size.to(dtype)
    n_treatments = n_treatments.to(dtype)

    # Validate inputs
    effect_size = torch.clamp(effect_size, min=1e-8)
    n_treatments = torch.clamp(n_treatments, min=3.0)

    # Initial approximation using chi-square power formula
    sqrt2 = math.sqrt(2.0)
    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2

    # Initial guess based on simplified chi-square formula
    # Rearranging: n ≈ (z_α + z_β)² * k * (k+1) / (12 * effect_size)
    n_init = (
        ((z_alpha + z_beta) ** 2)
        * n_treatments
        * (n_treatments + 1)
        / (12 * effect_size)
    )
    n_init = torch.clamp(n_init, min=5.0)

    # Iterative refinement
    n_current = n_init
    for _ in range(12):
        # Calculate current power
        current_power = friedman_test_power(
            effect_size, n_current, n_treatments, alpha=alpha
        )

        # Adjust sample size based on power gap
        power_gap = torch.clamp(power - current_power, min=-0.4, max=0.4)
        adjustment = 1.0 + 1.2 * power_gap
        n_current = torch.clamp(n_current * adjustment, min=5.0, max=1e6)

    # Round up to nearest integer
    n_out = torch.ceil(n_current)

    if out is not None:
        out.copy_(n_out)
        return out
    return n_out
