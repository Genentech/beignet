from typing import Callable

from torch import Tensor


def _long_range_interaction(
    fn: Callable[..., Tensor],
    displacement_fn: Callable[[Tensor, Tensor], Tensor],
):
    raise NotImplementedError
