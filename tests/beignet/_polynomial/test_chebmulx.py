import beignet.polynomial
import torch


def test_chebmulx():
    torch.testing.assert_close(beignet.polynomial.chebmulx([0]), [0])
    torch.testing.assert_close(beignet.polynomial.chebmulx([1]), [0, 1])
    for i in range(1, 5):
        ser = [0] * i + [1]
        tgt = [0] * (i - 1) + [0.5, 0, 0.5]
        torch.testing.assert_close(beignet.polynomial.chebmulx(ser), tgt)
