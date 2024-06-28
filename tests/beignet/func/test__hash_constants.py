import pytest
import torch
from beignet.func._molecular_dynamics._partition.__hash_constants import _hash_constants


def test_hash_constants_uniform_grid():
    spatial_dimensions = 3
    cells_per_side = torch.tensor([2])
    expected_output = torch.tensor([[1, 2, 4]], dtype=torch.int32)
    assert torch.equal(
        _hash_constants(spatial_dimensions, cells_per_side), expected_output
    )


def test_hash_constants_invalid_input_size():
    spatial_dimensions = 3
    cells_per_side = torch.tensor([2, 3])
    with pytest.raises(ValueError):
        _hash_constants(spatial_dimensions, cells_per_side)


def test_hash_constants_zero_dimensions():
    spatial_dimensions = 3
    cells_per_side = torch.tensor([])
    with pytest.raises(ValueError):
        _hash_constants(spatial_dimensions, cells_per_side)
