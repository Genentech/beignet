import beignet.polynomial
import beignet.polynomial._evaluate_power_series_1d
import torch


def test_evaluate_power_series_1d():
    assert beignet.polynomial.evaluate_power_series_1d([], [1]).shape == (0,)

    x = torch.linspace(-1, 1, 50)

    y = []

    for index in range(5):
        y = [*y, x**index]

    for index in range(5):
        torch.testing.assert_close(
            beignet.polynomial.evaluate_power_series_1d(
                x,
                [0] * index + [1],
            ),
            y[index],
        )

    torch.testing.assert_close(
        beignet.polynomial.evaluate_power_series_1d(
            x,
            [0, -1, 0, 1],
        ),
        x * (x**2 - 1),
    )

    for index in range(3):
        dims = [2] * index
        x = torch.zeros(dims)
        torch.testing.assert_close(
            beignet.polynomial.evaluate_power_series_1d(x, [1]).shape, dims
        )
        torch.testing.assert_close(
            beignet.polynomial.evaluate_power_series_1d(x, [1, 0]).shape, dims
        )
        torch.testing.assert_close(
            beignet.polynomial.evaluate_power_series_1d(x, [1, 0, 0]).shape, dims
        )
