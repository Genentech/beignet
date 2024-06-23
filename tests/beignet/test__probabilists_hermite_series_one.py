import beignet.polynomial
import torch


def test_probabilists_hermite_series_one():
    torch.testing.assert_close(
        beignet.polynomial.probabilists_hermite_series_one,
        torch.tensor([1]),
    )
