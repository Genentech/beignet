import math

import torch
from torch import Tensor

from ._kolmogorov_smirnov_test_power import kolmogorov_smirnov_test_power


def kolmogorov_smirnov_test_sample_size(
    effect_size: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Required sample size for Kolmogorov-Smirnov test.

    Calculates the sample size needed to achieve desired power for detecting
    a specified effect size in a Kolmogorov-Smirnov test.

    Parameters
    ----------
    effect_size : Tensor
        Effect size (maximum difference between distributions) in range [0, 1].
    power : float, default=0.8
        Desired statistical power.
    alpha : float, default=0.05
        Significance level.
    alternative : {"two-sided", "greater", "less"}, default="two-sided"
        Alternative hypothesis direction.

    Returns
    -------
    output : Tensor
        Required sample size (rounded up to nearest integer).

    Examples
    --------
    >>> effect_size = torch.tensor(0.3)
    >>> kolmogorov_smirnov_test_sample_size(effect_size)
    tensor(50.0)

    Notes
    -----
    This function uses an iterative approach to find the sample size that
    achieves the desired power. The calculation is based on the asymptotic
    distribution of the Kolmogorov-Smirnov test statistic.

    For two-sample tests, the returned value represents the harmonic mean
    sample size. For equal group sizes n₁ = n₂ = n, use n = result.
    For unequal sizes, solve: 2*n₁*n₂/(n₁+n₂) = result.
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))

    # Ensure floating point dtype
    dtype = effect_size.dtype
    if not dtype.is_floating_point:
        dtype = torch.float32
    effect_size = effect_size.to(dtype)

    # Validate inputs
    effect_size = torch.clamp(effect_size, min=1e-8, max=1.0)

    # Normalize alternative
    alt = alternative.lower()
    if alt in {"larger", "greater", ">"}:
        alt = "greater"
    elif alt in {"smaller", "less", "<"}:
        alt = "less"
    elif alt != "two-sided":
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    # Initial approximation based on asymptotic theory
    if alt == "two-sided":
        if alpha == 0.05:
            c_alpha = 1.36
        elif alpha == 0.01:
            c_alpha = 1.63
        else:
            c_alpha = torch.sqrt(-0.5 * torch.log(torch.tensor(alpha / 2, dtype=dtype)))
    else:
        if alpha == 0.05:
            c_alpha = 1.22
        elif alpha == 0.01:
            c_alpha = 1.52
        else:
            c_alpha = torch.sqrt(-0.5 * torch.log(torch.tensor(alpha, dtype=dtype)))

    sqrt2 = math.sqrt(2.0)
    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2

    # Initial guess: n ≈ (c_α / effect_size)²
    # This is based on the requirement that D_critical ≈ effect_size for power
    n_init = (c_alpha / effect_size) ** 2
    n_init = torch.clamp(n_init, min=10.0)

    # For power calculation, we need a more sophisticated approach
    # Add adjustment factor based on desired power
    adjustment = 1.0 + 0.5 * (z_beta / c_alpha) ** 2
    n_init = n_init * adjustment

    # Iterative refinement
    n_current = n_init
    for _ in range(12):
        # Calculate current power
        current_power = kolmogorov_smirnov_test_power(
            effect_size, n_current, alpha=alpha, alternative=alternative
        )

        # Adjust sample size based on power gap
        power_gap = torch.clamp(power - current_power, min=-0.4, max=0.4)
        # Use more aggressive adjustment for KS test as it converges slowly
        adjustment = 1.0 + 1.5 * power_gap
        n_current = torch.clamp(n_current * adjustment, min=10.0, max=1e6)

    # Round up to nearest integer
    n_out = torch.ceil(n_current)

    if out is not None:
        out.copy_(n_out)
        return out
    return n_out
