import beignet.polynomial
import numpy
import torch


def test_chebyshev_nodes_2():
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._chebpts2.chebyshev_nodes_2, 1.5
    )
    numpy.testing.assert_raises(
        ValueError, beignet.polynomial._chebpts2.chebyshev_nodes_2, 1
    )
    torch.testing.assert_close(
        beignet.polynomial._chebpts2.chebyshev_nodes_2(2), [-1, 1]
    )
    torch.testing.assert_close(
        beignet.polynomial._chebpts2.chebyshev_nodes_2(3), [-1, 0, 1]
    )
    torch.testing.assert_close(
        beignet.polynomial._chebpts2.chebyshev_nodes_2(4), [-1, -0.5, 0.5, 1]
    )
    torch.testing.assert_close(
        beignet.polynomial._chebpts2.chebyshev_nodes_2(5),
        [-1.0, -0.707106781187, 0, 0.707106781187, 1.0],
    )
