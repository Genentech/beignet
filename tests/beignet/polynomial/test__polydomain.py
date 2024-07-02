import torch
from beignet.polynomial import polydomain


def test_polydomain():
    torch.testing.assert_close(
        polydomain,
        torch.tensor([-1.0, 1.0]),
        check_dtype=False,
    )
