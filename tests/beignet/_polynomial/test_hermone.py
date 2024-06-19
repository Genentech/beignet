import beignet.polynomial
import beignet.polynomial._hermone
import torch


def test_hermone():
    torch.testing.assert_close(
        beignet.polynomial._hermone.hermone,
        torch.tensor([1]),
    )
