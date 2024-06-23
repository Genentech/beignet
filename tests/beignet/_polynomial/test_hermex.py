import beignet.polynomial
import torch


def test_hermex():
    torch.testing.assert_close(beignet.polynomial.probabilists_hermite_series_x, [0, 1])
