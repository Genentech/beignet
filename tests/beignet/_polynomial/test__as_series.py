import beignet.polynomial
import numpy


def test__as_series():
    numpy.testing.assert_raises(ValueError, beignet.polynomial._as_series, [[]])
    numpy.testing.assert_raises(ValueError, beignet.polynomial._as_series, [[[1, 2]]])
    numpy.testing.assert_raises(ValueError, beignet.polynomial._as_series, [[1], ["a"]])

    types = ["i", "d"]
    for i in range(len(types)):
        for j in range(i):
            [resi, resj] = beignet.polynomial._as_series(
                [(numpy.ones(1, types[i])), (numpy.ones(1, types[j]))]
            )
            numpy.testing.assert_(resi.dtype.char == resj.dtype.char)
            numpy.testing.assert_(resj.dtype.char == types[i])
