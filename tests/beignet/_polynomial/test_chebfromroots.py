import beignet.polynomial
import beignet.polynomial._chebtrim
import numpy


def test_chebfromroots():
    numpy.testing.assert_almost_equal(
        beignet.polynomial._chebtrim.chebtrim(
            beignet.polynomial.chebfromroots([]), tolerance=1e-6
        ),
        [1],
    )
    for i in range(1, 5):
        roots = numpy.cos(numpy.linspace(-numpy.pi, 0, 2 * i + 1)[1::2])
        tgt = [0] * i + [1]
        numpy.testing.assert_almost_equal(
            beignet.polynomial._chebtrim.chebtrim(
                beignet.polynomial.chebfromroots(roots) * 2 ** (i - 1),
                tolerance=1e-6,
            ),
            beignet.polynomial._chebtrim.chebtrim(tgt, tolerance=1e-6),
        )
