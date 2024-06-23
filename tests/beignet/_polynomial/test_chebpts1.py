import beignet.polynomial
import beignet.polynomial._chebpts1
import numpy
import torch


def test_chebpts1():
    numpy.testing.assert_raises(ValueError, beignet.polynomial._chebpts1.chebpts1, 1.5)
    numpy.testing.assert_raises(ValueError, beignet.polynomial._chebpts1.chebpts1, 0)
    torch.testing.assert_close(beignet.polynomial._chebpts1.chebpts1(1), [0])
    torch.testing.assert_close(
        beignet.polynomial._chebpts1.chebpts1(2),
        [-0.70710678118654746, 0.70710678118654746],
    )
    torch.testing.assert_close(
        beignet.polynomial._chebpts1.chebpts1(3),
        [-0.86602540378443871, 0, 0.86602540378443871],
    )
    torch.testing.assert_close(
        beignet.polynomial._chebpts1.chebpts1(4),
        [-0.9238795325, -0.3826834323, 0.3826834323, 0.9238795325],
    )
