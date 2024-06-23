import beignet.polynomial
import beignet.polynomial._physicists_hermite_series_zero
import torch


def test_physicists_hermite_series_zero():
    torch.testing.assert_close(
        beignet.polynomial.physicists_hermite_series_zero,
        torch.tensor([0]),
    )
