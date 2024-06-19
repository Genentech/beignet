import beignet.polynomial
import beignet.polynomial._lagzero
import torch


def test_lagzero():
    torch.testing.assert_close(
        beignet.polynomial._lagzero.lagzero,
        torch.tensor([0]),
    )
