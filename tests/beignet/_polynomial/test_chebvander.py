import beignet.polynomial
import beignet.polynomial._chebval
import beignet.polynomial._chebvander
import numpy


def test_chebvander():
    x = numpy.arange(3)
    v = beignet.polynomial._chebvander.chebvander(x, 3)
    numpy.testing.assert_(v.shape == (3, 4))
    for i in range(4):
        coef = [0] * i + [1]
        numpy.testing.assert_almost_equal(
            v[..., i], beignet.polynomial._chebval.chebval(x, coef)
        )

    x = numpy.array([[1, 2], [3, 4], [5, 6]])
    v = beignet.polynomial._chebvander.chebvander(x, 3)
    numpy.testing.assert_(v.shape == (3, 2, 4))
    for i in range(4):
        coef = [0] * i + [1]
        numpy.testing.assert_almost_equal(
            v[..., i], beignet.polynomial._chebval.chebval(x, coef)
        )
