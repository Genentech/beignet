import beignet.polynomial
import beignet.polynomial._legendre_series_domain
import torch


def test_legdomain():
    torch.testing.assert_close(
        beignet.polynomial._legdomain.legendre_series_domain,
        torch.tensor([-1.0, 1.0]),
    )
