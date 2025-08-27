from torch import Tensor

from beignet.statistics._cohens_d import cohens_d


def hedges_g(
    input: Tensor,
    other: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""

    Parameters
    ----------
    input : Tensor

    other : Tensor

    out : Tensor | None

    Returns
    -------
    Tensor
    """

    cohens_d_value = cohens_d(input, other, pooled=True)

    sample_size_group_1 = input.shape[-1]
    sample_size_group_2 = other.shape[-1]

    degrees_of_freedom = sample_size_group_1 + sample_size_group_2 - 2

    correction = 1.0 - 3.0 / (4.0 * degrees_of_freedom - 1.0)

    output = cohens_d_value * correction

    if out is not None:
        out.copy_(output)

        return out

    return output
