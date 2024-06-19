import beignet.polynomial
import beignet.polynomial._legval
import beignet.polynomial._legvander
import numpy


def test_legvander():
    x = numpy.arange(3)
    v = beignet.polynomial._legvander.legvander(x, 3)
    numpy.testing.assert_(v.shape == (3, 4))
    for i in range(4):
        coef = [0] * i + [1]
        numpy.testing.assert_almost_equal(
            v[..., i], beignet.polynomial._legval.legval(x, coef)
        )

    x = numpy.array([[1, 2], [3, 4], [5, 6]])
    v = beignet.polynomial._legvander.legvander(x, 3)
    numpy.testing.assert_(v.shape == (3, 2, 4))
    for i in range(4):
        coef = [0] * i + [1]
        numpy.testing.assert_almost_equal(
            v[..., i], beignet.polynomial._legval.legval(x, coef)
        )

    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._legvander.legvander, (1, 2, 3), -1
    )
