import math
from typing import Optional

import torch
from torch import Tensor


def _segment_sum(
    input: Tensor,
    indexes: Tensor,
    n: Optional[int] = None,
    **kwargs,
) -> Tensor:
    if indexes.ndim == 1:
        indexes = torch.repeat_interleave(indexes, math.prod([*input.shape[1:]])).view(
            *[indexes.shape[0], *input.shape[1:]]
        )

    if n is None:
        n = max([*indexes]) + 1

    output = torch.zeros(n, *input.shape[1:], device=input.device)

    return output.scatter_add(0, indexes, input.to(torch.float32)).to(**kwargs)
