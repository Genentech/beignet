import beignet.polynomial
import beignet.polynomial._physicists_hermite_series_one
import torch


def test_hermone():
    torch.testing.assert_close(
        beignet.polynomial.physicists_hermite_series_one,
        torch.tensor([1]),
    )
