import beignet.polynomial
import numpy


def test_polyval():
    numpy.testing.assert_equal(beignet.polynomial.polyval([], [1]).size, 0)

    x = numpy.linspace(-1, 1)
    y = [x**i for i in range(5)]
    for i in range(5):
        tgt = y[i]
        numpy.testing.assert_almost_equal(
            beignet.polynomial.polyval(x, [0] * i + [1]), tgt
        )
    tgt = x * (x**2 - 1)
    numpy.testing.assert_almost_equal(beignet.polynomial.polyval(x, [0, -1, 0, 1]), tgt)

    for i in range(3):
        dims = [2] * i
        x = numpy.zeros(dims)
        numpy.testing.assert_equal(beignet.polynomial.polyval(x, [1]).shape, dims)
        numpy.testing.assert_equal(beignet.polynomial.polyval(x, [1, 0]).shape, dims)
        numpy.testing.assert_equal(beignet.polynomial.polyval(x, [1, 0, 0]).shape, dims)

    mask = [False, True, False]
    numpy.testing.assert_array_equal(
        numpy.polyval([7, 5, 3], numpy.ma.array([1, 2, 3], mask=mask)).mask, mask
    )

    class C(numpy.ndarray):
        pass

    numpy.testing.assert_equal(
        type(numpy.polyval([2, 3, 4], numpy.array([1, 2, 3]).view(C))), C
    )
