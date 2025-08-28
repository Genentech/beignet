"""Cramer's V effect size functional metric."""

import torch
from torch import Tensor


def cramers_v(observed: Tensor, expected: Tensor) -> Tensor:
    """
    Compute Cramer's V effect size for chi-square test.

    Parameters
    ----------
    observed : Tensor
        Observed frequencies in contingency table.
    expected : Tensor
        Expected frequencies in contingency table.

    Returns
    -------
    Tensor
        Cramer's V effect size.

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics.functional.statistics import cramers_v
    >>> observed = torch.tensor([[10, 20], [30, 40]], dtype=torch.float)
    >>> expected = torch.tensor([[15, 15], [35, 35]], dtype=torch.float)
    >>> effect_size = cramers_v(observed, expected)
    """
    # Calculate chi-square statistic
    chi_square = torch.sum((observed - expected) ** 2 / expected)

    # Get dimensions
    n = torch.sum(observed)
    min_dim = min(observed.shape[-2] - 1, observed.shape[-1] - 1)

    return torch.sqrt(chi_square / (n * min_dim))
