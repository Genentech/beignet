import functools

import optree
import torch
from torch import Tensor


@functools.singledispatch
def _degrees_of_freedom_metric(positions: Tensor) -> int:
    # util.check_custom_simulation_type(position)

    def _fn(accumulator: Tensor, x: Tensor) -> int:
        return accumulator + torch.numel(x)

    return optree.tree_reduce(_fn, positions, 0)
