import beignet.polynomial
import torch


def test_polymulx():
    torch.testing.assert_close(beignet.polynomial.polymulx([0]), [0])
    torch.testing.assert_close(beignet.polynomial.polymulx([1]), [0, 1])
    for i in range(1, 5):
        torch.testing.assert_close(
            beignet.polynomial.polymulx([0] * i + [1]), [0] * (i + 1) + [1]
        )
