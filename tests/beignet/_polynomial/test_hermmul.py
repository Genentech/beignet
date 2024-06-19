import beignet.polynomial
import beignet.polynomial._hermmul
import beignet.polynomial._hermval
import numpy


def test_hermmul():
    x = numpy.linspace(-3, 3, 100)

    for i in range(5):
        pol1 = [0] * i + [1]
        val1 = beignet.polynomial._hermval.hermval(x, pol1)
        for j in range(5):
            msg = f"At i={i}, j={j}"
            pol2 = [0] * j + [1]
            val2 = beignet.polynomial._hermval.hermval(x, pol2)
            pol3 = beignet.polynomial._hermmul.hermmul(pol1, pol2)
            val3 = beignet.polynomial._hermval.hermval(x, pol3)
            numpy.testing.assert_(len(pol3) == i + j + 1, msg)
            numpy.testing.assert_almost_equal(val3, val1 * val2, err_msg=msg)
