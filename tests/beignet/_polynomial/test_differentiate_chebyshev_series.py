import beignet.polynomial
import beignet.polynomial._differentiate_chebyshev_series
import beignet.polynomial._integrate_chebyshev_series
import beignet.polynomial._trim_chebyshev_series
import numpy
import torch


def test_differentiate_chebyshev_series():
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial.differentiate_chebyshev_series, [0], 0.5
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial.differentiate_chebyshev_series, [0], -1
    )

    for i in range(5):
        tgt = [0] * i + [1]
        torch.testing.assert_close(
            beignet.polynomial.trim_chebyshev_series(
                beignet.polynomial.differentiate_chebyshev_series(tgt, m=0),
                tolerance=1e-6,
            ),
            beignet.polynomial.trim_chebyshev_series(tgt, tolerance=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            torch.testing.assert_close(
                beignet.polynomial.trim_chebyshev_series(
                    beignet.polynomial.differentiate_chebyshev_series(
                        beignet.polynomial.integrate_chebyshev_series(tgt, m=j),
                        m=j,
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial.trim_chebyshev_series(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            torch.testing.assert_close(
                beignet.polynomial.trim_chebyshev_series(
                    beignet.polynomial.differentiate_chebyshev_series(
                        beignet.polynomial.integrate_chebyshev_series(tgt, m=j, scl=2),
                        m=j,
                        scl=0.5,
                    ),
                    tolerance=1e-6,
                ),
                beignet.polynomial.trim_chebyshev_series(tgt, tolerance=1e-6),
            )

    c2d = numpy.random.random((3, 4))

    torch.testing.assert_close(
        beignet.polynomial.differentiate_chebyshev_series(c2d, axis=0),
        numpy.vstack(
            [beignet.polynomial.differentiate_chebyshev_series(c) for c in c2d.T]
        ).T,
    )

    torch.testing.assert_close(
        beignet.polynomial.differentiate_chebyshev_series(c2d, axis=1),
        numpy.vstack(
            [beignet.polynomial.differentiate_chebyshev_series(c) for c in c2d]
        ),
    )
