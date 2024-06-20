import beignet.polynomial
import beignet.polynomial._chebder
import beignet.polynomial._chebint
import beignet.polynomial._trim_chebyshev_series
import numpy
import torch


def test_chebder():
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial._chebder.chebder, [0], 0.5
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._chebder.chebder, [0], -1
    )

    for i in range(5):
        tgt = [0] * i + [1]
        torch.testing.assert_close(
            beignet.polynomial._chebtrim.trim_chebyshev_series(
                beignet.polynomial._chebder.chebder(tgt, m=0), tolerance=1e-6
            ),
            beignet.polynomial._chebtrim.trim_chebyshev_series(tgt, tolerance=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            numpy.testing.assert_almost_equal(
                beignet.polynomial._chebtrim.trim_chebyshev_series(
                    beignet.polynomial._chebder.chebder(
                        beignet.polynomial._chebint.chebint(tgt, m=j), m=j
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial._chebtrim.trim_chebyshev_series(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            numpy.testing.assert_almost_equal(
                beignet.polynomial._chebtrim.trim_chebyshev_series(
                    beignet.polynomial._chebder.chebder(
                        beignet.polynomial._chebint.chebint(tgt, m=j, scl=2),
                        m=j,
                        scl=0.5,
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial._chebtrim.trim_chebyshev_series(tgt, tolerance=1e-6),
            )

    c2d = numpy.random.random((3, 4))

    numpy.testing.assert_almost_equal(
        beignet.polynomial._chebder.chebder(c2d, axis=0),
        numpy.vstack([beignet.polynomial._chebder.chebder(c) for c in c2d.T]).T,
    )

    numpy.testing.assert_almost_equal(
        beignet.polynomial._chebder.chebder(c2d, axis=1),
        numpy.vstack([beignet.polynomial._chebder.chebder(c) for c in c2d]),
    )
