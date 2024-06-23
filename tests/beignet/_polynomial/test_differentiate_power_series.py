import beignet.polynomial
import numpy
import torch


def test_differentiate_power_series():
    numpy.testing.assert_raises(
        TypeError, beignet.polynomial.differentiate_power_series, [0], 0.5
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial.differentiate_power_series, [0], -1
    )

    for i in range(5):
        tgt = [0] * i + [1]
        torch.testing.assert_close(
            beignet.polynomial.trim_power_series(
                beignet.polynomial.differentiate_power_series(tgt, order=0),
                tolerance=1e-6,
            ),
            beignet.polynomial.trim_power_series(tgt, tolerance=1e-6),
        )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = beignet.polynomial.differentiate_power_series(
                beignet.polynomial.integrate_power_series(tgt, m=j), order=j
            )

            torch.testing.assert_close(
                beignet.polynomial.trim_power_series(res, tolerance=1e-6),
                beignet.polynomial.trim_power_series(tgt, tolerance=1e-6),
            )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = beignet.polynomial.differentiate_power_series(
                beignet.polynomial.integrate_power_series(tgt, m=j, scl=2),
                order=j,
                scale=0.5,
            )

            torch.testing.assert_close(
                beignet.polynomial.trim_power_series(res, tolerance=1e-6),
                beignet.polynomial.trim_power_series(tgt, tolerance=1e-6),
            )

    c2d = numpy.random.random((3, 4))

    tgt = numpy.vstack(
        [beignet.polynomial.differentiate_power_series(c) for c in c2d.T]
    ).T

    torch.testing.assert_close(
        beignet.polynomial.differentiate_power_series(c2d, axis=0), tgt
    )

    tgt = torch.vstack([beignet.polynomial.differentiate_power_series(c) for c in c2d])

    torch.testing.assert_close(
        beignet.polynomial.differentiate_power_series(
            c2d,
            axis=1,
        ),
        tgt,
    )
