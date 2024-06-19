import beignet.polynomial
import numpy
import torch


def test__c_series_to_z_series():
    for i in range(5):
        torch.testing.assert_close(
            beignet.polynomial._c_series_to_z_series(
                numpy.array([2] + [1] * i, numpy.float64)
            ),
            numpy.array([0.5] * i + [2] + [0.5] * i, numpy.float64),
        )
