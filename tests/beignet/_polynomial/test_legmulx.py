import beignet.polynomial
import torch


def test_legmulx():
    torch.testing.assert_close(beignet.polynomial.legmulx([0]), [0])
    torch.testing.assert_close(beignet.polynomial.legmulx([1]), [0, 1])
    for i in range(1, 5):
        tmp = 2 * i + 1
        ser = [0] * i + [1]
        tgt = [0] * (i - 1) + [i / tmp, 0, (i + 1) / tmp]
        torch.testing.assert_close(beignet.polynomial.legmulx(ser), tgt)
