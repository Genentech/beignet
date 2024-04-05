from torch import Tensor


def invert_rotation_vector(
    input: Tensor,
) -> Tensor:
    r"""
    Invert rotation vectors.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        Rotation vectors.

    Returns
    -------
    inverted_rotation_vectors : Tensor, shape (..., 3)
        Inverted rotation vectors.
    """
    return -input
