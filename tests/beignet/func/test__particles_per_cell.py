import pytest
import torch
from torch import Tensor
from unittest.mock import patch

from beignet.func._molecular_dynamics._partition.__particles_per_cell import _particles_per_cell


@patch('beignet.func._molecular_dynamics._partition.__particles_per_cell._cell_dimensions')
@patch('beignet.func._molecular_dynamics._partition.__particles_per_cell._hash_constants')
@patch('beignet.func._molecular_dynamics._partition.__particles_per_cell._segment_sum')
def test_particles_per_cell(mock_segment_sum, mock_hash_constants, mock_cell_dimensions):
    positions = torch.tensor([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]])
    size = torch.tensor([3.0, 3.0])
    minimum_size = 1.0

    mock_cell_dimensions.return_value = (size, torch.tensor([1.0, 1.0]), torch.tensor([3, 3]), 9)
    mock_hash_constants.return_value = torch.tensor([1, 3], dtype=torch.int32)
    mock_segment_sum.return_value = torch.tensor([1, 0, 0, 1, 0, 0, 0, 0, 1], dtype=torch.int32)

    expected_output = torch.tensor([1, 0, 0, 1, 0, 0, 0, 0, 1], dtype=torch.int32)

    output = _particles_per_cell(positions, size, minimum_size)

    assert torch.equal(output, expected_output)
    mock_cell_dimensions.assert_called_once_with(2, size, minimum_size)

    assert mock_hash_constants.call_count == 1
    args, _ = mock_hash_constants.call_args
    assert args[0] == 2
    assert torch.equal(args[1], torch.tensor([3, 3]))

    mock_segment_sum.assert_called_once()

if __name__ == "__main__":
    pytest.main()