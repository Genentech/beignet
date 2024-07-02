import functools
import operator

from torch import Tensor

from .__nth_slice import _nth_slice


def _vandermonde(functions, input: Tensor, degrees: Tensor) -> Tensor:
    n_dims = len(functions)

    if n_dims != len(input):
        raise ValueError

    if n_dims != len(degrees):
        raise ValueError

    if n_dims == 0:
        raise ValueError

    # produce the vandermonde matrix for each dimension, placing the last
    # axis of each in an independent trailing axis of the output
    vander_arrays = (
        functions[i](input[i], degrees[i])[(...,) + _nth_slice(i, n_dims)]
        for i in range(n_dims)
    )

    # we checked this wasn't empty already, so no `initial` needed
    return functools.reduce(operator.mul, vander_arrays)
