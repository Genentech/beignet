import beignet.polynomial
import beignet.polynomial._physicists_hermite_series_weight
import torch.testing


def test_physicists_hermite_series_weight():
    x = torch.linspace(-5, 5, 11)
    torch.testing.assert_close(
        beignet.polynomial.physicists_hermite_series_weight(x),
        torch.exp(-(x**2)),
    )
