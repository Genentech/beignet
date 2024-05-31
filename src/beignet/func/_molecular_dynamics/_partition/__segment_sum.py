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
    """
   Computes the sum of segments of a tensor along the first dimension.

   Parameters
   ----------
   input : Tensor
       A tensor containing the input values to be summed.
   indexes : Tensor
       A 1D tensor containing the segment indexes for summation.
       Should have the same length as the first dimension of the `input` tensor.
   n : Optional[int], optional
       The number of segments, by default `n` is set to `max(indexes) + 1`.

   Returns
   -------
   Tensor
       A tensor where each entry contains the sum of the corresponding segment
       from the `input` tensor.
   """
    if indexes.ndim == 1:
        indexes = torch.repeat_interleave(indexes, math.prod([*input.shape[1:]])).view(
            *[indexes.shape[0], *input.shape[1:]]
        )

    if input.size(0) != indexes.size(0):
        raise ValueError("The length of the indexes tensor must match the size of the first dimension of the input tensor.")

    if n is None:
        n = indexes.max().item() + 1

    output = torch.zeros(n, *input.shape[1:], device=input.device)

    return output.scatter_add(0, indexes, input.to(torch.float32)).to(**kwargs)
