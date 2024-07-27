from torch import Tensor


def pairwise_displacement(input: Tensor, other: Tensor) -> Tensor:
    r"""Compute a matrix of pairwise displacements given two sets of positions.

    Parameters
    ----------
    input : Tensor
        Vector of positions

    other : Tensor
        Vector of positions

    Returns:
    -------
    output : Tensor, shape [spatial_dimensions]
        Matrix of displacements
    """
    if len(input.shape) != 1:
        message = (
            "Can only compute displacements between vectors. To compute "
            "displacements between sets of vectors use vmap or TODO."
        )
        raise ValueError(message)

    if input.shape != other.shape:
        message = "Can only compute displacement between vectors of equal dimension."
        raise ValueError(message)

    return input - other
