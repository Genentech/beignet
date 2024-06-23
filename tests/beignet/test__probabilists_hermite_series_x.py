import beignet.polynomial
import torch


def test_probabilists_hermite_series_x():
    torch.testing.assert_close(
        beignet.polynomial.probabilists_hermite_series_x,
        torch.tensor([0, 1]),
    )
