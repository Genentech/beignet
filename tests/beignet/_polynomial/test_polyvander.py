import beignet.polynomial
import beignet.polynomial._polyval
import beignet.polynomial._polyvander
import numpy


def test_polyvander():
    x = numpy.arange(3)
    v = beignet.polynomial._polyvander.polyvander(x, 3)
    numpy.testing.assert_(v.shape == (3, 4))
    for i in range(4):
        numpy.testing.assert_almost_equal(
            v[..., i], beignet.polynomial._polyval.polyval(x, [0] * i + [1])
        )
    x = numpy.array([[1, 2], [3, 4], [5, 6]])
    v = beignet.polynomial._polyvander.polyvander(x, 3)
    numpy.testing.assert_(v.shape == (3, 2, 4))
    for i in range(4):
        numpy.testing.assert_almost_equal(
            v[..., i], beignet.polynomial._polyval.polyval(x, [0] * i + [1])
        )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._polyvander.polyvander, numpy.arange(3), -1
    )
