import beignet.polynomial
import numpy
import torch.testing


def test_probabilists_hermite_series_weight():
    x = torch.linspace(-5, 5, 11)

    torch.testing.assert_close(
        beignet.polynomial.probabilists_hermite_series_weight(x),
        numpy.exp(-0.5 * x**2),
    )
