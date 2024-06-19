import beignet.polynomial
import beignet.polynomial._polyx
import torch


def test_polyx():
    torch.testing.assert_close(
        beignet.polynomial._polyx.polyx,
        torch.tensor([0, 1]),
    )
