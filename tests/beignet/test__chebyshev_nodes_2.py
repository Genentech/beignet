import beignet.polynomial
import pytest
import torch


def test_chebyshev_nodes_2():
    with pytest.raises(ValueError):
        beignet.polynomial.chebyshev_nodes_2(torch.tensor([1.5]))

    with pytest.raises(ValueError):
        beignet.polynomial.chebyshev_nodes_2(torch.tensor([1]))

    torch.testing.assert_close(
        beignet.polynomial.chebyshev_nodes_2(
            torch.tensor([2]),
        ),
        torch.tensor(
            [-1, 1],
            dtype=torch.float32,
        ),
    )

    torch.testing.assert_close(
        beignet.polynomial.chebyshev_nodes_2(
            torch.tensor([3]),
        ),
        torch.tensor(
            [-1, 0, 1],
            dtype=torch.float32,
        ),
    )

    torch.testing.assert_close(
        beignet.polynomial.chebyshev_nodes_2(
            torch.tensor([4]),
        ),
        torch.tensor(
            [-1, -0.5, 0.5, 1],
            dtype=torch.float32,
        ),
    )

    torch.testing.assert_close(
        beignet.polynomial.chebyshev_nodes_2(
            torch.tensor([5]),
        ),
        torch.tensor(
            [-1.0, -0.707106781187, 0, 0.707106781187, 1.0],
            dtype=torch.float32,
        ),
    )
