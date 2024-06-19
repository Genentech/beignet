import beignet.polynomial
import torch


def test_hermx():
    torch.testing.assert_close(
        beignet.polynomial.hermx,
        torch.tensor([0, 0.5]),
    )
