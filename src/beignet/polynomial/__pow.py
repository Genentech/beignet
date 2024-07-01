from typing import Callable

import torch
from torch import Tensor

from beignet.polynomial import _as_series


def _pow(
    func: Callable,
    input: Tensor,
    exponent: int | Tensor,
    maximum_exponent: int | Tensor,
) -> Tensor:
    [input] = _as_series([input])

    _exponent = int(exponent)

    if _exponent != exponent or _exponent < 0:
        raise ValueError

    if maximum_exponent is not None and _exponent > maximum_exponent:
        raise ValueError

    match _exponent:
        case 0:
            output = torch.tensor([1], dtype=input.dtype)
        case 1:
            output = input
        case _:
            output = torch.zeros(input.shape[0] * exponent, dtype=input.dtype)

            [output, input] = _as_series([output, input])

            if output.shape[0] > input.shape[0]:
                input = torch.concatenate(
                    [
                        input,
                        torch.zeros(
                            output.shape[0] - input.shape[0],
                            dtype=input.dtype,
                        ),
                    ],
                )

                output = output + input
            else:
                output = torch.concatenate(
                    [
                        output,
                        torch.zeros(
                            input.shape[0] - output.shape[0],
                            dtype=output.dtype,
                        ),
                    ]
                )

                output = input + output

            for _ in range(2, _exponent + 1):
                output = func(output, input, mode="same")

    return output
