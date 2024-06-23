import beignet.polynomial
import numpy
import torch

hermite_e_polynomial_coefficients = [
    (torch.tensor([1], dtype=torch.float64)),
    (torch.tensor([0, 1], dtype=torch.float64)),
    (torch.tensor([-1, 0, 1], dtype=torch.float64)),
    (torch.tensor([0, -3, 0, 1], dtype=torch.float64)),
    (torch.tensor([3, 0, -6, 0, 1], dtype=torch.float64)),
    (torch.tensor([0, 15, 0, -10, 0, 1], dtype=torch.float64)),
    (torch.tensor([-15, 0, 45, 0, -15, 0, 1], dtype=torch.float64)),
    (torch.tensor([0, -105, 0, 105, 0, -21, 0, 1], dtype=torch.float64)),
    (torch.tensor([105, 0, -420, 0, 210, 0, -28, 0, 1], dtype=torch.float64)),
    (torch.tensor([0, 945, 0, -1260, 0, 378, 0, -36, 0, 1], dtype=torch.float64)),
]


def test_evaluate_probabilists_hermite_series_1d():
    x = numpy.random.random((3, 5)) * 2 - 1

    torch.testing.assert_close(
        beignet.polynomial.evaluate_probabilists_hermite_series_1d([], [1]).size,
        0,
    )

    x = numpy.linspace(-1, 1)
    y = [
        beignet.polynomial.evaluate_power_series_1d(x, c)
        for c in hermite_e_polynomial_coefficients
    ]
    for i in range(10):
        msg = f"At i={i}"
        tgt = y[i]
        res = beignet.polynomial.evaluate_probabilists_hermite_series_1d(
            x, [0] * i + [1]
        )
        numpy.testing.assert_almost_equal(res, tgt, err_msg=msg)

    for i in range(3):
        dims = [2] * i
        x = numpy.zeros(dims)
        torch.testing.assert_close(
            beignet.polynomial.evaluate_probabilists_hermite_series_1d(x, [1]).shape,
            dims,
        )
        torch.testing.assert_close(
            beignet.polynomial.evaluate_probabilists_hermite_series_1d(x, [1, 0]).shape,
            dims,
        )
        torch.testing.assert_close(
            beignet.polynomial.evaluate_probabilists_hermite_series_1d(
                x, [1, 0, 0]
            ).shape,
            dims,
        )
