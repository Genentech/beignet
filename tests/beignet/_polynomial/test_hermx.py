import beignet.polynomial
import beignet.polynomial._physicists_hermite_series_x
import torch


def test_hermx():
    torch.testing.assert_close(
        beignet.polynomial._hermx.physicists_hermite_series_x,
        torch.tensor([0, 0.5]),
    )
