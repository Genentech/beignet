import beignet.polynomial
import beignet.polynomial._legendre_series_zero
import torch


def test_legzero():
    torch.testing.assert_close(
        beignet.polynomial._legzero.legendre_series_zero,
        torch.tensor([0]),
    )
