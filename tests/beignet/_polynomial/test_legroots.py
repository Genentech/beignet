import beignet.polynomial
import beignet.polynomial._legtrim
import numpy


def test_legroots():
    numpy.testing.assert_almost_equal(beignet.polynomial.legroots([1]), [])
    numpy.testing.assert_almost_equal(beignet.polynomial.legroots([1, 2]), [-0.5])
    for i in range(2, 5):
        tgt = numpy.linspace(-1, 1, i)
        res = beignet.polynomial.legroots(beignet.polynomial.legfromroots(tgt))
        numpy.testing.assert_almost_equal(
            beignet.polynomial._legtrim.legtrim(res, tolerance=1e-6),
            beignet.polynomial._legtrim.legtrim(tgt, tolerance=1e-6),
        )
