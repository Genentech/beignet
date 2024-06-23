import beignet.polynomial
import torch


def test_physicists_hermite_series_line():
    torch.testing.assert_close(
        beignet.polynomial.physicists_hermite_series_line(3, 4),
        torch.tensor([3, 2], dtype=torch.float32),
    )
