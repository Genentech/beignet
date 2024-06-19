import beignet.polynomial
import torch


def test_hermemulx():
    torch.testing.assert_close(beignet.polynomial.hermemulx([0]), [0])
    torch.testing.assert_close(beignet.polynomial.hermemulx([1]), [0, 1])
    for i in range(1, 5):
        ser = [0] * i + [1]
        tgt = [0] * (i - 1) + [i, 0, 1]
        torch.testing.assert_close(beignet.polynomial.hermemulx(ser), tgt)
