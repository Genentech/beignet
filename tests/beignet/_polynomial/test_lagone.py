import beignet.polynomial
import beignet.polynomial._lagone
import torch


def test_lagone():
    torch.testing.assert_close(
        beignet.polynomial._lagone.lagone,
        torch.tensor([1]),
    )
