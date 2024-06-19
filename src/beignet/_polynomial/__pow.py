import numpy

from .__as_series import _as_series


def _pow(func, input, exponent, maximum_exponent):
    [input] = _as_series([input])

    exponent = int(exponent)

    if exponent != exponent or exponent < 0:
        raise ValueError("Power must be a non-negative integer.")
    elif maximum_exponent is not None and exponent > maximum_exponent:
        raise ValueError("Power is too large")
    elif exponent == 0:
        return numpy.array([1], dtype=input.dtype)
    elif exponent == 1:
        return input
    else:
        output = input

        for _ in range(2, exponent + 1):
            output = func(output, input)

        return output
