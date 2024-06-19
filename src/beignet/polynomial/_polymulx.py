import torch

from .__as_series import _as_series


def polymulx(input):
    (input,) = _as_series([input])

    if len(input) == 1 and input[0] == 0:
        return input

    output = torch.empty(len(input) + 1, dtype=input.dtype)

    output[0] = input[0] * 0.0

    output[1:] = input

    return output
