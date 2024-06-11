import torch
from torch import Tensor


def _is_space_valid(space: Tensor) -> Tensor:
    r"""Check if the given space tensor is valid.

    Parameters:
    -----------
    space : Tensor
        The space tensor to be validated. This tensor can have 0, 1, or 2 dimensions.

    Returns:
    --------
    Tensor
        A tensor containing a single boolean value indicating whether the space is valid.

    Raises:
    -------
    ValueError
        If the space tensor has more than 2 dimensions.
    """
    if space.ndim == 0 or space.ndim == 1:
        return torch.tensor([True])

    if space.ndim == 2:
        return torch.tensor([torch.all(torch.triu(space) == space)])

    return torch.tensor([False])
