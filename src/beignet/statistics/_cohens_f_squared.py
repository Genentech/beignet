from torch import Tensor

from beignet.statistics import cohens_f


def cohens_f_squared(
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
    output = cohens_f(input, other) ** 2

    if out is not None:
        out.copy_(output)

        return out

    return output
