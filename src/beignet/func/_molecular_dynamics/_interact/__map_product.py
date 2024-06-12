from typing import Callable

import torch
from torch import Tensor


def _map_product(
    fn: Callable[[Tensor, Tensor], Tensor],
) -> Callable[[Tensor, Tensor], Tensor]:
    """

    Parameters
    ----------
    fn

    Returns
    -------

    """
    return torch.func.vmap(
        torch.func.vmap(
            fn,
            (0, None),
            0,
        ),
        (None, 0),
        0,
    )
