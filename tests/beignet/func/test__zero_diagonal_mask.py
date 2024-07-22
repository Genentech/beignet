import pytest
import torch

from beignet.func._interact import _zero_diagonal_mask


def test_zero_diagonal_mask_2d():
    matrix = torch.rand(5, 5)

    masked_matrix = _zero_diagonal_mask(matrix)

    assert torch.equal(
        torch.diagonal(masked_matrix, 0), torch.zeros(5)
    )


def test_zero_diagonal_mask_3d():
    matrix = torch.rand(5, 5, 5)

    masked_matrix = _zero_diagonal_mask(matrix)

    for i in range(matrix.shape[0]):
        assert torch.equal(
            masked_matrix[i][i], torch.zeros(5)
        )


def test_zero_diagonal_mask_rectangular_matrix():
    with pytest.raises(ValueError):
        _zero_diagonal_mask(torch.rand(3, 5, 5))


def test_zero_diagonal_mask_rank_4():
    with pytest.raises(ValueError):
        _zero_diagonal_mask(torch.rand(4, 4, 4, 4))
