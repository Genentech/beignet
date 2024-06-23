import beignet.polynomial
import torch


def test_hermezero():
    torch.testing.assert_close(
        beignet.polynomial.probabilists_hermite_series_zero,
        torch.tensor([0]),
    )
