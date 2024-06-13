import numpy

from ._trimseq import trimseq


def as_series(alist, trim=True):
    arrays = [numpy.array(a, ndmin=1, copy=False) for a in alist]

    if min([a.size for a in arrays]) == 0:
        raise ValueError("Coefficient array is empty")

    if any(a.ndim != 1 for a in arrays):
        raise ValueError("Coefficient array is not 1-d")

    if trim:
        arrays = [trimseq(a) for a in arrays]

    if any(a.dtype == numpy.dtype(object) for a in arrays):
        ret = []

        for a in arrays:
            if a.dtype != numpy.dtype(object):
                tmp = numpy.empty(len(a), dtype=numpy.dtype(object))

                tmp[:] = a[:]

                ret.append(tmp)
            else:
                ret.append(a.copy())
    else:
        try:
            dtype = numpy.common_type(*arrays)
        except Exception as e:
            raise ValueError("Coefficient arrays have no common type") from e

        ret = [numpy.array(a, copy=True, dtype=dtype) for a in arrays]

    return ret
