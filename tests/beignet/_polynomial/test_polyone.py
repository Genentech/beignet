import beignet.polynomial
import torch


def test_polyone():
    torch.testing.assert_close(
        beignet.polynomial.polyone,
        torch.tensor([1]),
    )
