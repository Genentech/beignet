import beignet.polynomial
import beignet.polynomial._chebfromroots
import beignet.polynomial._chebroots
import beignet.polynomial._chebtrim
import numpy


def test_chebroots():
    numpy.testing.assert_almost_equal(beignet.polynomial._chebroots.chebroots([1]), [])
    numpy.testing.assert_almost_equal(
        beignet.polynomial._chebroots.chebroots([1, 2]), [-0.5]
    )
    for i in range(2, 5):
        tgt = numpy.linspace(-1, 1, i)
        numpy.testing.assert_almost_equal(
            beignet.polynomial._chebtrim.chebtrim(
                beignet.polynomial._chebroots.chebroots(
                    beignet.polynomial._chebfromroots.chebfromroots(tgt)
                ),
                tolerance=1e-6,
            ),
            beignet.polynomial._chebtrim.chebtrim(tgt, tolerance=1e-6),
        )
