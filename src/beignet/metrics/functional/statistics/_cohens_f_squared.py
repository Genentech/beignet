"""Cohen's f² effect size functional metric."""

from torch import Tensor

import beignet.metrics.functional.statistics


def cohens_f_squared(
    groups: list[Tensor],
    *,
    out: Tensor | None = None,
) -> Tensor:
    """
    Compute Cohen's f² effect size from multiple sample groups.

    Cohen's f² is simply the square of Cohen's f.

    Parameters
    ----------
    groups : list[Tensor]
        List of sample groups, each of shape (..., N_i) where N_i is the sample size for group i.
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        The Cohen's f² values.
    """
    cohens_f_value = beignet.metrics.functional.statistics.cohens_f(groups)
    result = cohens_f_value**2

    if out is not None:
        out.copy_(result)
        return out

    return result
