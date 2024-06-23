import beignet.polynomial
import numpy
import torch


def test_power_series_vandermonde_3d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random([2, 3, 4])

    x1 = torch.from_numpy(x1)
    x2 = torch.from_numpy(x2)
    x3 = torch.from_numpy(x3)

    c = torch.from_numpy(c)

    torch.testing.assert_close(
        torch.dot(
            beignet.polynomial.power_series_vandermonde_3d(
                x1,
                x2,
                x3,
                [1, 2, 3],
            ),
            torch.flatten(c),
        ),
        beignet.polynomial.evaluate_power_series_3d(x1, x2, x3, c),
    )

    output = beignet.polynomial.power_series_vandermonde_3d(
        torch.tensor([x1]),
        torch.tensor([x2]),
        torch.tensor([x3]),
        [1, 2, 3],
    )

    assert output.shape == (1, 5, 24)
