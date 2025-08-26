import math

import torch
from torch import Tensor

from ._jonckheere_terpstra_test_power import jonckheere_terpstra_test_power


def jonckheere_terpstra_test_sample_size(
    effect_size: Tensor,
    groups: Tensor | int,
    power: float = 0.8,
    alpha: float = 0.05,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Required sample size per group for Jonckheere-Terpstra test.

    Calculates the sample size needed per group to achieve desired power
    for detecting a specified ordered trend effect in a Jonckheere-Terpstra test.

    Parameters
    ----------
    effect_size : Tensor
        Effect size representing the standardized trend across ordered groups.
    groups : Tensor or int
        Number of ordered groups to compare.
    power : float, default=0.8
        Desired statistical power.
    alpha : float, default=0.05
        Significance level (one-tailed).

    Returns
    -------
    output : Tensor
        Required sample size per group (rounded up to nearest integer).

    Examples
    --------
    >>> effect_size = torch.tensor(0.5)
    >>> groups = 4
    >>> jonckheere_terpstra_test_sample_size(effect_size, groups)
    tensor(15.0)

    Notes
    -----
    This function assumes equal group sizes and uses an iterative approach
    to find the sample size that achieves the desired power.

    The Jonckheere-Terpstra test is particularly useful when:
    - You have a priori expectations about the ordering of groups
    - The groups represent ordered categories (e.g., dose levels)
    - You want more power than the Kruskal-Wallis test for detecting trends
    """
    effect_size = torch.atleast_1d(torch.as_tensor(effect_size))
    groups = torch.atleast_1d(torch.as_tensor(groups))

    # Ensure floating point dtype
    dtype = (
        torch.float64
        if (effect_size.dtype == torch.float64 or groups.dtype == torch.float64)
        else torch.float32
    )
    effect_size = effect_size.to(dtype)
    groups = groups.to(dtype)

    # Validate inputs
    effect_size = torch.clamp(effect_size, min=1e-8)
    groups = torch.clamp(groups, min=3.0)

    # Initial approximation
    sqrt2 = math.sqrt(2.0)
    z_alpha = torch.erfinv(torch.tensor(1 - alpha, dtype=dtype)) * sqrt2
    z_beta = torch.erfinv(torch.tensor(power, dtype=dtype)) * sqrt2

    # Initial guess based on normal approximation
    # For trend tests, the required n is typically smaller than for omnibus tests
    n_init = ((z_alpha + z_beta) / effect_size) ** 2 / groups
    n_init = torch.clamp(n_init, min=5.0)

    # Iterative refinement
    n_current = n_init
    for _ in range(12):
        # Create equal sample sizes
        sample_sizes = n_current.unsqueeze(-1).expand(
            *n_current.shape, int(groups.max().item())
        )

        # Calculate current power
        current_power = jonckheere_terpstra_test_power(
            effect_size, sample_sizes, alpha=alpha
        )

        # Adjust sample size based on power gap
        power_gap = torch.clamp(power - current_power, min=-0.4, max=0.4)
        adjustment = 1.0 + 1.3 * power_gap
        n_current = torch.clamp(n_current * adjustment, min=5.0, max=1e6)

    # Round up to nearest integer
    n_out = torch.ceil(n_current)

    if out is not None:
        out.copy_(n_out)
        return out
    return n_out
