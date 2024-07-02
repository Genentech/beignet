import torch
from beignet.polynomial import polyone


def test_polyone():
    torch.testing.assert_close(
        polyone,
        torch.tensor([1.0]),
        check_dtype=False,
    )
