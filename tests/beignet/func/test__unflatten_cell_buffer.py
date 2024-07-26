import pytest
import torch

from beignet.func._partition import _unflatten_cell_buffer


def test__unflatten_cell_buffer_cells_per_side_is_scalar():
    buffer = torch.arange(60).reshape(10, 2, 3)
    cells_per_side = torch.tensor(5)
    dim = 1

    assert torch.equal(
        _unflatten_cell_buffer(buffer, cells_per_side, dim),
        torch.arange(60).reshape(5, 2, 2, 3),
    )


def test__unflatten_cell_buffer_cells_per_side_is_1d_tensor():
    buffer = torch.arange(60).reshape(10, 2, 3)
    cells_per_side = torch.tensor([5])
    dim = 1

    expected_result = torch.arange(60).reshape(5, 2, 2, 3)

    result = _unflatten_cell_buffer(buffer, cells_per_side, dim)

    assert torch.equal(
        result, expected_result
    ), f"Expected: {expected_result}, but got: {result}"


@pytest.fixture
def expected_tensor():
    return torch.tensor(
        [
            [[0, 1], [2, 3], [4, 5], [6, 7]],
            [[8, 9], [10, 11], [12, 13], [14, 15]],
            [[16, 17], [18, 19], [20, 21], [22, 23]],
        ]
    )


def test__unflatten_cell_buffer_cells_per_side_is_1d_tensor_length_2(expected_tensor):
    buffer = torch.arange(24)
    cells_per_side = torch.tensor([4, 3])
    dim = 2

    expected_shape = (3, 4, 2)

    result = _unflatten_cell_buffer(buffer, cells_per_side, dim)

    assert torch.equal(
        result, expected_tensor
    ), f"Expected: {expected_tensor}, but got: {result}"
    assert result.shape == expected_shape


def test__unflatten_cell_buffer_cells_per_side_2d_length_2(expected_tensor):
    buffer = torch.arange(24)
    cells_per_side = torch.tensor([[4, 3]])
    dim = 2

    expected_shape = (3, 4, 2)

    result = _unflatten_cell_buffer(buffer, cells_per_side, dim)

    assert torch.equal(
        result, expected_tensor
    ), f"Expected: {expected_tensor}, but got: {result}"
    assert result.shape == expected_shape


def test__unflatten_cell_buffer_cells_per_side_2d_with_values_in_each_dimension(
    expected_tensor,
):
    buffer = torch.arange(24)
    cells_per_side = torch.tensor([[4, 3], [2, 3]])
    dim = 2

    expected_shape = (3, 4, 2)

    result = _unflatten_cell_buffer(buffer, cells_per_side, dim)

    assert torch.equal(
        result, expected_tensor
    ), f"Expected: {expected_tensor}, but got: {result}"
    assert result.shape == expected_shape
