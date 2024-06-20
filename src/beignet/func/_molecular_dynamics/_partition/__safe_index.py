from torch import Tensor


def safe_index(array: Tensor, indices: Tensor) -> Tensor:
    """
    Safely index into a tensor, clamping out-of-bounds indices to the nearest valid index.

    Parameters:
    array (Tensor): The tensor to index.
    indices (Tensor): The indices to use for indexing.

    Returns:
    Tensor: The resulting tensor after indexing.
    """
    max_index = array.shape[0] - 1

    clamped_indices = indices.clamp(0, max_index)

    result = array[clamped_indices]

    return result
