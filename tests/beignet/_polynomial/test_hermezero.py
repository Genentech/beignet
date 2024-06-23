import beignet.polynomial
import beignet.polynomial._probabilists_hermite_series_zero
import torch


def test_hermezero():
    torch.testing.assert_close(
        beignet.polynomial._hermezero.probabilists_hermite_series_zero,
        torch.tensor([0]),
    )
