import beignet.polynomial
import numpy


def test__c_series_to_z_series():
    for i in range(5):
        numpy.testing.assert_equal(
            beignet.polynomial._c_series_to_z_series(
                numpy.array([2] + [1] * i, numpy.float64)
            ),
            numpy.array([0.5] * i + [2] + [0.5] * i, numpy.float64),
        )
