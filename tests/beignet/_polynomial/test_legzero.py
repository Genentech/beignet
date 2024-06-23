import beignet.polynomial
import beignet.polynomial._legzero
import torch


def test_legzero():
    torch.testing.assert_close(
        beignet.polynomial._legzero.legendre_series_zero,
        torch.tensor([0]),
    )
