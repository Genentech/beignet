from typing import Callable

from torch import Tensor


def _angle_interaction(
    fn: Callable[..., Tensor],
    displacement_fn: Callable[[Tensor, Tensor], Tensor],
):
    raise NotImplementedError
