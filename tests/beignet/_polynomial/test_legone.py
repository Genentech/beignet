import beignet.polynomial
import beignet.polynomial._legone
import torch


def test_legone():
    torch.testing.assert_close(
        beignet.polynomial._legone.legendre_series_one,
        torch.tensor([1]),
    )
