import torch
from beignet.polynomial import polyx


def test_polyx():
    torch.testing.assert_close(
        polyx,
        torch.tensor([0.0, 1.0]),
        check_dtype=False,
    )
