import math

import torch
from torch import Tensor

from ._kruskal_wallis_test_power import kruskal_wallis_test_power


def kruskal_wallis_test_sample_size(
    effect_size: Tensor,
    k: Tensor | int,
    power: float = 0.8,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Required sample size per group for Kruskal-Wallis H test.

    Calculates the sample size needed per group to achieve desired power
    for detecting a specified effect size in a Kruskal-Wallis test.

    Parameters
    ----------
    effect_size : Tensor
        Effect size (variance of group location parameters divided by error variance).
    k : Tensor or int
        Number of groups to compare.
    power : float, default=0.8
        Desired statistical power.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    output : Tensor
        Required sample size per group (rounded up to nearest integer).

    Examples
    --------
    >>> effect_size = torch.tensor(0.3)
    >>> k = 3
    >>> kruskal_wallis_test_sample_size(effect_size, k)
    tensor(15.0)

    Notes
    -----
    This function assumes equal group sizes and uses an iterative approach
    to find the sample size that achieves the desired power. The calculation
    is based on the chi-square approximation to the Kruskal-Wallis H statistic.

    For unequal group sizes, use the power function directly with different
    sample size vectors to find the optimal allocation.
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    k = torch.atleast_1d(torch.as_tensor(k))

    # Ensure floating point dtype
    if effect_size.dtype.is_floating_point and k.dtype.is_floating_point:
        if effect_size.dtype == torch.float64 or k.dtype == torch.float64:
            dtype = torch.float64
        else:
            dtype = torch.float32
    else:
        dtype = torch.float32
    effect_size = effect_size.to(dtype)
    k = k.to(dtype)

    # Validate inputs
    effect_size = torch.clamp(effect_size, min=1e-8)
    k = torch.clamp(k, min=3.0)

    # Initial approximation using chi-square power formula
    sqrt2 = math.sqrt(2.0)
    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2

    df = k - 1

    # Initial guess based on simplified formula
    # For chi-square test: n ≈ (z_α + z_β)² / (effect_size * df)
    n_init = ((z_alpha + z_beta) ** 2) / (effect_size * df)
    n_init = torch.clamp(n_init, min=5.0)

    # Iterative refinement
    n_current = n_init
    for _ in range(12):
        # Create sample sizes tensor (equal allocation)
        sample_sizes = n_current.unsqueeze(-1).expand(
            *n_current.shape, int(k.max().item())
        )

        # Calculate current power
        current_power = kruskal_wallis_test_power(
            effect_size, sample_sizes, alpha=alpha
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
