import beignet.polynomial
import pytest
import torch


def test_chebyshev_nodes_1():
    with pytest.raises(ValueError):
        beignet.polynomial.chebyshev_nodes_1(
            torch.tensor([1.5]),
        )

    with pytest.raises(ValueError):
        beignet.polynomial.chebyshev_nodes_1(
            torch.tensor([0]),
        )

    torch.testing.assert_close(
        beignet.polynomial.chebyshev_nodes_1(torch.tensor([1])),
        torch.tensor([0], dtype=torch.float32),
    )

    torch.testing.assert_close(
        beignet.polynomial.chebyshev_nodes_1(torch.tensor([2])),
        torch.tensor([-0.70710678118654746, 0.70710678118654746]),
    )

    torch.testing.assert_close(
        beignet.polynomial.chebyshev_nodes_1(torch.tensor([3])),
        torch.tensor([-0.86602540378443871, 0, 0.86602540378443871]),
    )

    torch.testing.assert_close(
        beignet.polynomial.chebyshev_nodes_1(torch.tensor([4])),
        torch.tensor([-0.9238795325, -0.3826834323, 0.3826834323, 0.9238795325]),
    )
