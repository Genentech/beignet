import beignet.polynomial
import numpy
import torch


def test_power_series_vandermonde_3d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = torch.randn([2, 3, 4])

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
        [x1],
        [x2],
        [x3],
        [1, 2, 3],
    )

    assert output.shape == (1, 5, 24)
