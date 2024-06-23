import beignet.polynomial
import numpy
import pytest
import torch


def test_differentiate_chebyshev_series():
    with pytest.raises(TypeError):
        beignet.polynomial.differentiate_chebyshev_series(
            torch.tensor([0]),
            0.5,
        )

    with pytest.raises(ValueError):
        beignet.polynomial.differentiate_chebyshev_series(
            torch.tensor([0]),
            -1,
        )

    for index in range(5):
        torch.testing.assert_close(
            beignet.polynomial.trim_chebyshev_series(
                beignet.polynomial.differentiate_chebyshev_series(
                    torch.tensor([0] * index + [1]),
                    m=0,
                ),
                tolerance=0.000001,
            ),
            beignet.polynomial.trim_chebyshev_series(
                torch.tensor([0] * index + [1]),
                tolerance=0.000001,
            ),
        )

    for j in range(5):
        for k in range(2, 5):
            torch.testing.assert_close(
                beignet.polynomial.trim_chebyshev_series(
                    beignet.polynomial.differentiate_chebyshev_series(
                        beignet.polynomial.integrate_chebyshev_series(
                            torch.tensor([0] * j + [1]),
                            m=k,
                        ),
                        m=k,
                    ),
                    tolerance=0.000001,
                ),
                beignet.polynomial.trim_chebyshev_series(
                    torch.tensor([0] * j + [1]),
                    tolerance=0.000001,
                ),
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
                    tolerance=0.000001,
                ),
                beignet.polynomial.trim_chebyshev_series(tgt, tolerance=0.000001),
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
