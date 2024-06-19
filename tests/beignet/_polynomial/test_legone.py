import beignet.polynomial
import beignet.polynomial._legone
import torch


def test_legone():
    torch.testing.assert_close(
        beignet.polynomial._legone.legone,
        torch.tensor([1]),
    )
