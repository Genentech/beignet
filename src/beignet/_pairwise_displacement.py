from torch import Tensor


def pairwise_displacement(Ra: Tensor, Rb: Tensor) -> Tensor:
    """Compute a matrix of pairwise displacements given two sets of positions.

    Args:
      Ra: Vector of positions; `Tensor(shape=[spatial_dim])`.
      Rb: Vector of positions; `Tensor(shape=[spatial_dim])`.

    Returns:
      Matrix of displacements; `Tensor(shape=[spatial_dim])`.
    """
    if len(Ra.shape) != 1:
        msg = (
            "Can only compute displacements between vectors. To compute "
            "displacements between sets of vectors use vmap or TODO."
        )
        raise ValueError(msg)

    if Ra.shape != Rb.shape:
        msg = "Can only compute displacement between vectors of equal dimension."
        raise ValueError(msg)

    return Ra - Rb
