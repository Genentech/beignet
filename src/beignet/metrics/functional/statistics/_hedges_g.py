"""Hedges' g effect size functional metric."""

from torch import Tensor

from ._cohens_d import cohens_d


def hedges_g(group1: Tensor, group2: Tensor) -> Tensor:
    """
    Compute Hedges' g effect size between two groups.

    Parameters
    ----------
    group1 : Tensor
        Samples from the first group.
    group2 : Tensor
        Samples from the second group.

    Returns
    -------
    Tensor
        Hedges' g effect size.

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics.functional.statistics import hedges_g
    >>> group1 = torch.randn(20)
    >>> group2 = torch.randn(20) + 0.5
    >>> effect_size = hedges_g(group1, group2)
    """
    cohens_d_value = cohens_d(group1, group2, pooled=True)
    n1 = group1.shape[-1]
    n2 = group2.shape[-1]
    df = n1 + n2 - 2
    correction_factor = 1 - (3 / (4 * df - 1))
    return cohens_d_value * correction_factor
