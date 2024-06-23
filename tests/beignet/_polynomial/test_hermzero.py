import beignet.polynomial
import beignet.polynomial._physicists_hermite_series_zero
import torch


def test_hermzero():
    torch.testing.assert_close(
        beignet.polynomial._hermzero.physicists_hermite_series_zero,
        torch.tensor([0]),
    )
