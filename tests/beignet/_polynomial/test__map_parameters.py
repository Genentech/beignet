import beignet.polynomial
import numpy


def test__map_parameters():
    numpy.testing.assert_almost_equal(
        beignet.polynomial._map_parameters([0, 4], [1, 3]), [1, 0.5]
    )
    numpy.testing.assert_almost_equal(
        beignet.polynomial._map_parameters([0 - 1j, 2 + 1j], [-2, 2]), [-1 + 1j, 1 - 1j]
    )
