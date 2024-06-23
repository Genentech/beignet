import beignet.polynomial
import beignet.polynomial._laguerre_series_domain
import torch


def test_lagdomain():
    torch.testing.assert_close(
        beignet.polynomial._lagdomain.laguerre_series_domain,
        torch.tensor([0.0, 1.0]),
    )
