import beignet.polynomial
import torch


def test_hermone():
    torch.testing.assert_close(
        beignet.polynomial.hermone,
        torch.tensor([1]),
    )
