from torch import Tensor

from beignet.statistics._cohens_d import cohens_d


def hedges_g(
    group1: Tensor,
    group2: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    group1 : Tensor
        Group1 parameter.
    group2 : Tensor
        Group2 parameter.
    out : Tensor | None
        Output tensor.

    Returns
    -------
    Tensor
        Computed statistic.
    """

    cohens_d_value = cohens_d(group1, group2, pooled=True)

    sample_size_group_1 = group1.shape[-1]
    sample_size_group_2 = group2.shape[-1]

    degrees_of_freedom = sample_size_group_1 + sample_size_group_2 - 2

    correction = 1.0 - 3.0 / (4.0 * degrees_of_freedom - 1.0)

    output = cohens_d_value * correction

    if out is not None:
        out.copy_(output)
        return out

    return output
