import beignet.polynomial
import beignet.polynomial._hermx
import torch


def test_hermx():
    torch.testing.assert_close(
        beignet.polynomial._hermx.hermx,
        torch.tensor([0, 0.5]),
    )
