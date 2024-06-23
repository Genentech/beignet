import beignet.polynomial
import torch


def test_physicists_hermite_series_x():
    torch.testing.assert_close(
        beignet.polynomial.physicists_hermite_series_x,
        torch.tensor([0, 0.5]),
    )
