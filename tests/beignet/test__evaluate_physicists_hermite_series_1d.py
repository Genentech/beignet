import beignet.polynomial
import numpy
import torch

hermite_polynomial_coefficients = [
    (torch.tensor([1])),
    (torch.tensor([0, 2])),
    (torch.tensor([-2, 0, 4])),
    (torch.tensor([0, -12, 0, 8])),
    (torch.tensor([12, 0, -48, 0, 16])),
    (torch.tensor([0, 120, 0, -160, 0, 32])),
    (torch.tensor([-120, 0, 720, 0, -480, 0, 64])),
    (torch.tensor([0, -1680, 0, 3360, 0, -1344, 0, 128])),
    (torch.tensor([1680, 0, -13440, 0, 13440, 0, -3584, 0, 256])),
    (torch.tensor([0, 30240, 0, -80640, 0, 48384, 0, -9216, 0, 512])),
]


def test_evaluate_physicists_hermite_series_1d():
    torch.testing.assert_close(
        beignet.polynomial.evaluate_physicists_hermite_series_1d([], [1]).size,
        0,
    )

    x = numpy.linspace(-1, 1)
    y = [
        beignet.polynomial.evaluate_power_series_1d(x, c)
        for c in hermite_polynomial_coefficients
    ]
    for i in range(10):
        msg = f"At i={i}"
        tgt = y[i]
        res = beignet.polynomial.evaluate_physicists_hermite_series_1d(x, [0] * i + [1])
        numpy.testing.assert_almost_equal(res, tgt, err_msg=msg)

    for i in range(3):
        dims = [2] * i
        x = numpy.zeros(dims)
        torch.testing.assert_close(
            beignet.polynomial.evaluate_physicists_hermite_series_1d(x, [1]).shape,
            dims,
        )
        torch.testing.assert_close(
            beignet.polynomial.evaluate_physicists_hermite_series_1d(x, [1, 0]).shape,
            dims,
        )
        torch.testing.assert_close(
            beignet.polynomial.evaluate_physicists_hermite_series_1d(
                x, [1, 0, 0]
            ).shape,
            dims,
        )
