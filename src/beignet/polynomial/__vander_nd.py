import functools
import operator

import numpy

from .__nth_slice import _nth_slice


def _vander_nd(vander_fs, points, degrees):
    n_dims = len(vander_fs)
    if n_dims != len(points):
        raise ValueError(
            f"Expected {n_dims} dimensions of sample points, got {len(points)}"
        )
    if n_dims != len(degrees):
        raise ValueError(f"Expected {n_dims} dimensions of degrees, got {len(degrees)}")
    if n_dims == 0:
        raise ValueError("Unable to guess a dtype or shape when no points are given")

    points = tuple(numpy.array(tuple(points), copy=False) + 0.0)

    vander_arrays = (
        vander_fs[i](points[i], degrees[i])[(...,) + _nth_slice(i, n_dims)]
        for i in range(n_dims)
    )

    return functools.reduce(operator.mul, vander_arrays)
