from typing import Iterable, Optional, Union

import torch
from torch import Tensor


def _safe_sum(
    x: Tensor,
    dim: Optional[Union[Iterable[int], int]] = None,
    keepdim: bool = False,
):
    match x:
        case _ if x.is_complex():
            promoted_dtype = torch.complex128
        case _ if x.is_floating_point():
            promoted_dtype = torch.float64
        case _:
            promoted_dtype = torch.int64

    summation = torch.sum(x, dim=dim, dtype=promoted_dtype, keepdim=keepdim)

    return summation.to(dtype=x.dtype)
