import torch
from beignet.polynomial import polyzero


def test_polyzero():
    torch.testing.assert_close(
        polyzero,
        torch.tensor([0.0]),
        check_dtype=False,
    )
