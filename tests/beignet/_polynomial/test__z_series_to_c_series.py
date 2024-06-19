import beignet.polynomial
import beignet.polynomial.__z_series_to_c_series
import numpy
import torch


def test__z_series_to_c_series():
    for index in range(5):
        torch.testing.assert_close(
            beignet.polynomial.__z_series_to_c_series._z_series_to_c_series(
                numpy.array(
                    [0.5] * index + [2] + [0.5] * index,
                    dtype=numpy.float64,
                ),
            ),
            numpy.array(
                [2] + [1] * index,
                dtype=numpy.float64,
            ),
        )
