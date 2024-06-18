import beignet.polynomial
import numpy
import torch


def test__as_series():
    numpy.testing.assert_raises(ValueError, beignet.polynomial._as_series, [[]])
    numpy.testing.assert_raises(ValueError, beignet.polynomial._as_series, [[[1, 2]]])
    numpy.testing.assert_raises(ValueError, beignet.polynomial._as_series, [[1], ["a"]])

    a, b = beignet.polynomial._as_series(
        [
            torch.rand([8], dtype=torch.float32),
            torch.rand([8], dtype=torch.float64),
        ]
    )

    assert a.dtype == b.dtype
