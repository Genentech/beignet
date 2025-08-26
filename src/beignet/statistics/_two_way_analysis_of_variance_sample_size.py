import math

import torch
from torch import Tensor

from ._two_way_analysis_of_variance_power import two_way_analysis_of_variance_power


def two_way_analysis_of_variance_sample_size(
    effect_size: Tensor,
    levels_factor_a: Tensor,
    levels_factor_b: Tensor,
    power: float = 0.8,
    alpha: float = 0.05,
    effect_type: str = "main_a",
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Required sample size per cell for two-way ANOVA.

    Calculates the sample size needed per cell to achieve desired power
    for testing main effects or interaction in two-way ANOVA.

    Parameters
    ----------
    effect_size : Tensor
        Cohen's f for the specific effect being tested.
    levels_factor_a : Tensor
        Number of levels for Factor A.
    levels_factor_b : Tensor
        Number of levels for Factor B.
    power : float, default=0.8
        Desired statistical power.
    alpha : float, default=0.05
        Significance level.
    effect_type : {"main_a", "main_b", "interaction"}, default="main_a"
        Which effect to test.

    Returns
    -------
    output : Tensor
        Required sample size per cell (rounded up to nearest integer).

    Examples
    --------
    >>> effect_size = torch.tensor(0.25)  # Medium effect
    >>> levels_factor_a = torch.tensor(2)
    >>> levels_factor_b = torch.tensor(3)
    >>> two_way_analysis_of_variance_sample_size(effect_size, levels_factor_a, levels_factor_b)
    tensor(18.0)

    Notes
    -----
    This function uses an iterative approach to find the sample size per cell
    that achieves the desired power for the specified effect in two-way ANOVA.

    Total sample size = sample_size_per_cell × levels_factor_a × levels_factor_b
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    levels_factor_a = torch.atleast_1d(torch.as_tensor(levels_factor_a))
    levels_factor_b = torch.atleast_1d(torch.as_tensor(levels_factor_b))

    # Ensure floating point dtype
    dtypes = [effect_size.dtype, levels_factor_a.dtype, levels_factor_b.dtype]
    if any(dt == torch.float64 for dt in dtypes):
        dtype = torch.float64
    else:
        dtype = torch.float32

    effect_size = effect_size.to(dtype)
    levels_factor_a = levels_factor_a.to(dtype)
    levels_factor_b = levels_factor_b.to(dtype)

    # Validate inputs
    effect_size = torch.clamp(effect_size, min=1e-8)
    levels_factor_a = torch.clamp(levels_factor_a, min=2.0)
    levels_factor_b = torch.clamp(levels_factor_b, min=2.0)

    # Initial approximation
    sqrt2 = math.sqrt(2.0)
    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2

    # Degrees of freedom for the effect
    if effect_type == "main_a":
        df_effect = levels_factor_a - 1
    elif effect_type == "main_b":
        df_effect = levels_factor_b - 1
    elif effect_type == "interaction":
        df_effect = (levels_factor_a - 1) * (levels_factor_b - 1)
    else:
        raise ValueError("effect_type must be 'main_a', 'main_b', or 'interaction'")

    # Initial guess based on chi-square approximation
    # λ = N * f²  and  (z_α + z_β)² ≈ λ / df_effect
    lambda_approx = ((z_alpha + z_beta) * torch.sqrt(df_effect)) ** 2
    n_per_cell_init = lambda_approx / (
        effect_size**2 * levels_factor_a * levels_factor_b
    )
    n_per_cell_init = torch.clamp(n_per_cell_init, min=5.0)

    # Iterative refinement
    n_current = n_per_cell_init
    for _ in range(12):
        # Calculate current power
        current_power = two_way_analysis_of_variance_power(
            effect_size,
            n_current,
            levels_factor_a,
            levels_factor_b,
            alpha=alpha,
            effect_type=effect_type,
        )

        # Adjust sample size based on power gap
        power_gap = torch.clamp(power - current_power, min=-0.4, max=0.4)
        adjustment = 1.0 + 1.1 * power_gap
        n_current = torch.clamp(n_current * adjustment, min=5.0, max=1e5)

    # Round up to nearest integer
    n_out = torch.ceil(n_current)

    if out is not None:
        out.copy_(n_out)
        return out
    return n_out
