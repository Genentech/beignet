import beignet.polynomial
import torch


def test_legendre_series_domain():
    torch.testing.assert_close(
        beignet.polynomial.legendre_series_domain,
        torch.tensor([-1.0, 1.0]),
    )
