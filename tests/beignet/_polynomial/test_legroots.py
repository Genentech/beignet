import beignet.polynomial
import numpy


def test_legroots():
    numpy.testing.assert_almost_equal(beignet.polynomial.legroots([1]), [])
    numpy.testing.assert_almost_equal(beignet.polynomial.legroots([1, 2]), [-0.5])
    for i in range(2, 5):
        tgt = numpy.linspace(-1, 1, i)
        res = beignet.polynomial.legroots(beignet.polynomial.legfromroots(tgt))
        numpy.testing.assert_almost_equal(
            beignet.polynomial.legtrim(res, tolerance=1e-6),
            beignet.polynomial.legtrim(tgt, tolerance=1e-6),
        )
