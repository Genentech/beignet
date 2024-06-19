import numpy

from .__vander_nd import _vander_nd


def _vander_nd_flat(vander_fs, points, degrees):
    v = _vander_nd(vander_fs, points, degrees)
    return numpy.reshape(v, v.shape[: -len(degrees)] + (-1,))
