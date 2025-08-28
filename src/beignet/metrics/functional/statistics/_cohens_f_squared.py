"""Cohen's f² effect size functional metric."""

from torch import Tensor

from ._cohens_f import cohens_f


def cohens_f_squared(groups: list[Tensor]) -> Tensor:
    """
    Compute Cohen's f² effect size for ANOVA.

    Parameters
    ----------
    groups : list of Tensor
        List of sample tensors for each group.

    Returns
    -------
    Tensor
        Cohen's f² effect size.

    Examples
    --------
    >>> import torch
    >>> from beignet.metrics.functional.statistics import cohens_f_squared
    >>> groups = [torch.randn(20), torch.randn(20) + 0.5, torch.randn(20) + 1.0]
    >>> effect_size = cohens_f_squared(groups)
    """
    f_value = cohens_f(groups)
    return f_value**2
