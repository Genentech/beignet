import beignet.polynomial
import torch


def test_hermmulx():
    torch.testing.assert_close(beignet.polynomial.hermmulx([0]), [0])
    torch.testing.assert_close(beignet.polynomial.hermmulx([1]), [0, 0.5])
    for i in range(1, 5):
        ser = [0] * i + [1]
        tgt = [0] * (i - 1) + [i, 0, 0.5]
        torch.testing.assert_close(beignet.polynomial.hermmulx(ser), tgt)
