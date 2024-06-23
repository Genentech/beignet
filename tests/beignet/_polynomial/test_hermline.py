import beignet.polynomial
import beignet.polynomial._physicists_hermite_series_line
import torch


def test_hermline():
    torch.testing.assert_close(
        beignet.polynomial._hermline.physicists_hermite_series_line(3, 4),
        torch.tensor([3, 2], dtype=torch.float32),
    )
