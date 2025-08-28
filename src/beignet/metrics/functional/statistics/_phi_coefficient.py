"""Phi coefficient functional metric."""

import torch
from torch import Tensor


def phi_coefficient(observed: Tensor, expected: Tensor) -> Tensor:
    """
    Compute Phi coefficient for 2x2 contingency table.

    Parameters
    ----------
    observed : Tensor
        Observed frequencies in 2x2 contingency table.
    expected : Tensor
        Expected frequencies in 2x2 contingency table.

    Returns
    -------
    Tensor
        Phi coefficient.

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics.functional.statistics import phi_coefficient
    >>> observed = torch.tensor([[10, 20], [30, 40]], dtype=torch.float)
    >>> expected = torch.tensor([[15, 15], [35, 35]], dtype=torch.float)
    >>> effect_size = phi_coefficient(observed, expected)
    """
    if observed.shape[-2] != 2 or observed.shape[-1] != 2:
        raise ValueError("Phi coefficient requires a 2x2 contingency table")

    # Calculate chi-square statistic
    chi_square = torch.sum((observed - expected) ** 2 / expected)
    n = torch.sum(observed)

    return torch.sqrt(chi_square / n)
