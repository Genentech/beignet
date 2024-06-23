import beignet.polynomial
import beignet.polynomial._physicists_hermite_series_one
import torch


def test_physicists_hermite_series_one():
    torch.testing.assert_close(
        beignet.polynomial.physicists_hermite_series_one,
        torch.tensor([1]),
    )
