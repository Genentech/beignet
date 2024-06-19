import beignet.polynomial
import torch


def test_polyx():
    torch.testing.assert_close(
        beignet.polynomial.polyx,
        torch.tensor([0, 1]),
    )
