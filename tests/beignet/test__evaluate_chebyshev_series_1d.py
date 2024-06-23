import beignet.polynomial
import numpy
import torch

chebyshev_polynomial_coefficients = [
    torch.tensor([1]),
    torch.tensor([0, 1]),
    torch.tensor([-1, 0, 2]),
    torch.tensor([0, -3, 0, 4]),
    torch.tensor([1, 0, -8, 0, 8]),
    torch.tensor([0, 5, 0, -20, 0, 16]),
    torch.tensor([-1, 0, 18, 0, -48, 0, 32]),
    torch.tensor([0, -7, 0, 56, 0, -112, 0, 64]),
    torch.tensor([1, 0, -32, 0, 160, 0, -256, 0, 128]),
    torch.tensor([0, 9, 0, -120, 0, 432, 0, -576, 0, 256]),
]


def test_evaluate_chebyshev_series_1d():
    torch.testing.assert_close(
        beignet.polynomial.evaluate_chebyshev_series_1d([], [1]).size, 0
    )

    x = numpy.linspace(-1, 1)
    y = [
        beignet.polynomial.evaluate_power_series_1d(x, c)
        for c in chebyshev_polynomial_coefficients
    ]
    for i in range(10):
        msg = f"At i={i}"
        tgt = y[i]
        res = beignet.polynomial.evaluate_chebyshev_series_1d(x, [0] * i + [1])
        numpy.testing.assert_almost_equal(res, tgt, err_msg=msg)

    for i in range(3):
        dims = [2] * i
        x = numpy.zeros(dims)
        torch.testing.assert_close(
            beignet.polynomial.evaluate_chebyshev_series_1d(x, [1]).shape, dims
        )
        torch.testing.assert_close(
            beignet.polynomial.evaluate_chebyshev_series_1d(x, [1, 0]).shape,
            dims,
        )
        torch.testing.assert_close(
            beignet.polynomial.evaluate_chebyshev_series_1d(x, [1, 0, 0]).shape,
            dims,
        )
