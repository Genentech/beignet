import beignet.polynomial
import numpy
import torch


def test_differentiate_probabilists_hermite_series():
    numpy.testing.assert_raises(
        TypeError,
        beignet.polynomial.differentiate_probabilists_hermite_series,
        [0],
        0.5,
    )
    numpy.testing.assert_raises(
        ValueError,
        beignet.polynomial.differentiate_probabilists_hermite_series,
        [0],
        -1,
    )

    for i in range(5):
        tgt = [0] * i + [1]
        res = beignet.polynomial.differentiate_probabilists_hermite_series(tgt, m=0)
        torch.testing.assert_close(
            beignet.polynomial.trim_probabilists_hermite_series(
                res, tolerance=0.000001
            ),
            beignet.polynomial.trim_probabilists_hermite_series(
                tgt, tolerance=0.000001
            ),
        )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = beignet.polynomial.differentiate_probabilists_hermite_series(
                beignet.polynomial.integrate_probabilists_hermite_series(tgt, m=j),
                m=j,
            )
            torch.testing.assert_close(
                beignet.polynomial.trim_probabilists_hermite_series(
                    res, tolerance=0.000001
                ),
                beignet.polynomial.trim_probabilists_hermite_series(
                    tgt, tolerance=0.000001
                ),
            )

    for i in range(5):
        for j in range(2, 5):
            tgt = [0] * i + [1]
            res = beignet.polynomial.differentiate_probabilists_hermite_series(
                beignet.polynomial.integrate_probabilists_hermite_series(
                    tgt, m=j, scl=2
                ),
                m=j,
                scl=0.5,
            )
            torch.testing.assert_close(
                beignet.polynomial.trim_probabilists_hermite_series(
                    res, tolerance=0.000001
                ),
                beignet.polynomial.trim_probabilists_hermite_series(
                    tgt, tolerance=0.000001
                ),
            )

    c2d = numpy.random.random((3, 4))

    tgt = numpy.vstack(
        [beignet.polynomial.differentiate_probabilists_hermite_series(c) for c in c2d.T]
    ).T

    res = beignet.polynomial.differentiate_probabilists_hermite_series(c2d, axis=0)

    torch.testing.assert_close(res, tgt)

    tgt = numpy.vstack(
        [beignet.polynomial.differentiate_probabilists_hermite_series(c) for c in c2d]
    )

    res = beignet.polynomial.differentiate_probabilists_hermite_series(c2d, axis=1)

    torch.testing.assert_close(
        res,
        tgt,
    )
