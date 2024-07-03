import torch
from torch import Tensor

from ._multiply_probabilists_hermite_polynomial import (
    multiply_probabilists_hermite_polynomial,
)


def probabilists_hermite_polynomial_power(
    input: Tensor,
    exponent: float | Tensor,
    maximum_exponent: float | Tensor = 16.0,
) -> Tensor:
    input = torch.atleast_1d(input)
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

            input = torch.atleast_1d(input)
            output = torch.atleast_1d(output)

            dtype = torch.promote_types(input.dtype, output.dtype)

            input = input.to(dtype)
            output = output.to(dtype)

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
                output = multiply_probabilists_hermite_polynomial(
                    output, input, mode="same"
                )
    return output
