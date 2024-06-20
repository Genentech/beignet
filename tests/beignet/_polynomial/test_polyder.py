import beignet.polynomial
import numpy
import torch


def test_polyder():
    numpy.testing.assert_raises(TypeError, beignet.polynomial.polyder, [0], 0.5)
    numpy.testing.assert_raises(ValueError, beignet.polynomial.polyder, [0], -1)

    for i in range(5):
        tgt = [0] * i + [1]
        torch.testing.assert_close(
            beignet.polynomial.trim_power_series(
                beignet.polynomial.polyder(tgt, order=0), tolerance=1e-6
            ),
            beignet.polynomial.trim_power_series(tgt, tolerance=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = beignet.polynomial.polyder(
                beignet.polynomial.polyint(tgt, m=j), order=j
            )

            torch.testing.assert_close(
                beignet.polynomial.trim_power_series(res, tolerance=1e-6),
                beignet.polynomial.trim_power_series(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = beignet.polynomial.polyder(
                beignet.polynomial.polyint(tgt, m=j, scl=2), order=j, scale=0.5
            )

            torch.testing.assert_close(
                beignet.polynomial.trim_power_series(res, tolerance=1e-6),
                beignet.polynomial.trim_power_series(tgt, tolerance=1e-6),
            )

    c2d = numpy.random.random((3, 4))

    tgt = numpy.vstack([beignet.polynomial.polyder(c) for c in c2d.T]).T

    torch.testing.assert_close(beignet.polynomial.polyder(c2d, axis=0), tgt)

    tgt = torch.vstack([beignet.polynomial.polyder(c) for c in c2d])

    torch.testing.assert_close(
        beignet.polynomial.polyder(
            c2d,
            axis=1,
        ),
        tgt,
    )
