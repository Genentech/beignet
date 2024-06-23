import beignet.polynomial
import beignet.polynomial._laguerre_series_domain
import torch


def test_laguerre_series_domain():
    torch.testing.assert_close(
        beignet.polynomial.laguerre_series_domain,
        torch.tensor([0.0, 1.0]),
    )
