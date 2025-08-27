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
    correction = 1.0 - 3.0 / (4.0 * (input.shape[-1] + other.shape[-1] - 2) - 1.0)

    output = cohens_d(input, other, pooled=True) * correction

    if out is not None:
        out.copy_(output)

        return out

    return output
