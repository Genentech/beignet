import beignet.polynomial
import beignet.polynomial.__as_series
import torch


def test__as_series():
    a, b = beignet.polynomial.__as_series._as_series(
        [
            torch.rand([8], dtype=torch.float32),
            torch.rand([8], dtype=torch.float64),
        ]
    )

    assert a.dtype == b.dtype
