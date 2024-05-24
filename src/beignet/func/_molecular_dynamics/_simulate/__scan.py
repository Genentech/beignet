from typing import Callable, Any

import torch
from torch import Tensor


def _scan(fn: Callable, carry: Any, indexes: Tensor):
    """
    Apply a function in a loop sequence over the input index tensor and accumulate the
    intermediate results.

    Parameters
    ----------
    fn : Callable
        The function to be applied iteratively. Takes the current "carry" value and the
        current value from `indexes`, and returns the next "carry" value and an output
        value.
    carry : Any
        The initial "carry" value that gets updated each time `fn` is applied.
    indexes : torch.Tensor
        The 1D index tensor over which to loop.

    Returns
    -------
    carry : Any
        The final "carry" value after applying `fn` to every item in `indexes`.
    ys : torch.Tensor
        Tensor of outputs after each application of `fn`.
    """
    ys = []

    for x in indexes:
        carry, y = fn(carry, x)

        ys.append(y)

    return carry, torch.tensor(ys)
