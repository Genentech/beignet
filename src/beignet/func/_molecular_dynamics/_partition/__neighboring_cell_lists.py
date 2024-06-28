from typing import Generator

import torch
from torch import Tensor


def _neighboring_cell_lists(
    dimension: int,
) -> Generator[Tensor, None, None]:
    for index in torch.cartesian_prod(*[torch.arange(3) for _ in range(dimension)]):
        yield index - 1
