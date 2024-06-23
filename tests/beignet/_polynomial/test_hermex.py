import beignet.polynomial
import beignet.polynomial._probabilists_hermite_series_x
import torch


def test_hermex():
    torch.testing.assert_close(
        beignet.polynomial._hermex.probabilists_hermite_series_x, [0, 1]
    )
