import beignet
import pytest
import torch


def test_chebyshev_zeros():
    with pytest.raises(ValueError):
        beignet.chebyshev_zeros(0)

    torch.testing.assert_close(
        beignet.chebyshev_zeros(1),
        torch.tensor([0.0]),
    )

    torch.testing.assert_close(
        beignet.chebyshev_zeros(2),
        torch.tensor([-0.70710678118654746, 0.70710678118654746]),
    )

    torch.testing.assert_close(
        beignet.chebyshev_zeros(3),
        torch.tensor([-0.86602540378443871, 0, 0.86602540378443871]),
    )

    torch.testing.assert_close(
        beignet.chebyshev_zeros(4),
        torch.tensor([-0.9238795325, -0.3826834323, 0.3826834323, 0.9238795325]),
    )
