import beignet.polynomial
import numpy
import torch


def test_chebyshev_nodes_1():
    numpy.testing.assert_raises(ValueError, beignet.polynomial.chebyshev_nodes_1, 1.5)
    numpy.testing.assert_raises(ValueError, beignet.polynomial.chebyshev_nodes_1, 0)
    torch.testing.assert_close(beignet.polynomial.chebyshev_nodes_1(1), [0])
    torch.testing.assert_close(
        beignet.polynomial.chebyshev_nodes_1(2),
        [-0.70710678118654746, 0.70710678118654746],
    )
    torch.testing.assert_close(
        beignet.polynomial.chebyshev_nodes_1(3),
        [-0.86602540378443871, 0, 0.86602540378443871],
    )
    torch.testing.assert_close(
        beignet.polynomial.chebyshev_nodes_1(4),
        [-0.9238795325, -0.3826834323, 0.3826834323, 0.9238795325],
    )
