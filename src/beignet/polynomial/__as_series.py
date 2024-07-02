from typing import List

import torch
from torch import Tensor


def _as_series(input: List[Tensor], trim: bool = False) -> List[Tensor]:
    outputs = []

    for i in input:
        output = torch.atleast_1d(i)

        if trim:
            if output.shape[0] != 0:
                j = 0

                for j in range(output.shape[0] - 1, -1, -1):
                    if output[j] != 0:
                        break

                output = output[: j + 1]

        outputs = [
            *outputs,
            output,
        ]

    dtype = outputs[0].dtype

    for output in outputs[1:]:
        dtype = torch.promote_types(dtype, output.dtype)

    for i, output in enumerate(outputs):
        if output.dtype != dtype:
            outputs[i] = output.to(dtype)

    return outputs
