import torch

from .__common_type import _common_type
from .__trim_sequence import _trim_sequence


def _as_series(inputs, trim=True):
    outputs = []

    for input in inputs:
        if input.ndim != 1:
            raise ValueError

        if input.size == 0:
            raise ValueError

        output = torch.ravel(input)

        if trim:
            output = _trim_sequence(output)

        outputs = [*outputs, output]

    dtype = _common_type(*outputs)

    for index, output in enumerate(outputs):
        outputs[index] = output.to(dtype)

    return outputs
