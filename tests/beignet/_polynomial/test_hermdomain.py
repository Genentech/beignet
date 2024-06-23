import beignet.polynomial
import torch


def test_hermdomain():
    torch.testing.assert_close(
        beignet.polynomial.physicists_hermite_series_domain,
        torch.tensor([-1.0, 1.0]),
    )
