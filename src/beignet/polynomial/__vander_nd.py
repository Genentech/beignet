import functools
import operator

import numpy


def _vander_nd(func, input, degrees):
    n_dims = len(func)
    if n_dims != len(input):
        raise ValueError(
            f"Expected {n_dims} dimensions of sample points, got {len(input)}"
        )
    if n_dims != len(degrees):
        raise ValueError(f"Expected {n_dims} dimensions of degrees, got {len(degrees)}")
    if n_dims == 0:
        raise ValueError("Unable to guess a dtype or shape when no points are given")

    input = tuple(numpy.asarray(tuple(input)) + 0.0)

    ys = []

    for index in range(n_dims):
        output = [None] * n_dims

        output[index] = slice(None)

        y = func[index](input[index], degrees[index])[(...,) + (*output,)]

        ys = [*ys, y]

    return functools.reduce(operator.mul, ys)
