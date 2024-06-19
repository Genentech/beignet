import functools
import operator

import numpy


def _vander_nd(func, input, degrees):
    n_dims = len(func)

    if n_dims != len(input):
        raise ValueError

    if n_dims != len(degrees):
        raise ValueError

    if n_dims == 0:
        raise ValueError

    input = tuple(numpy.asarray(tuple(input)) + 0.0)

    ys = []

    for index in range(n_dims):
        output = [None] * n_dims

        output[index] = slice(None)

        y = func[index](input[index], degrees[index])[(...,) + (*output,)]

        ys = [*ys, y]

    return functools.reduce(operator.mul, ys)
