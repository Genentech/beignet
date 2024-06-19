import operator

import numpy
import torch


def polyvander(input, degree):
    degree = operator.index(degree)

    if degree < 0:
        raise ValueError

    input = numpy.array(input, ndmin=1) + 0.0

    output = torch.empty([degree + 1, *input.shape], dtype=input.dtype)

    output[0] = input * 0.0 + 1.0

    if degree > 0:
        output[1] = input

        for i in range(2, degree + 1):
            output[i] = output[i - 1] * input

    output = torch.moveaxis(output, 0, -1)

    return output
