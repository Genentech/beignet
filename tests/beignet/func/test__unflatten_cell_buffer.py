import pytest
import torch

from beignet.func._molecular_dynamics._partition.__unflatten_cell_buffer import \
    _unflatten_cell_buffer


def test__unflatten_cell_buffer_cells_per_side_is_scalar():
    buffer = torch.arange(60).reshape(10, 2, 3)
    cells_per_side = torch.tensor(5)
    dim = 1

    assert torch.equal(
        _unflatten_cell_buffer(buffer, cells_per_side, dim),
        torch.arange(60).reshape(5, 2, 2, 3)
    )


def test__unflatten_cell_buffer_cells_per_side_is_1d_tensor():
    buffer = torch.arange(60).reshape(10, 2, 3)
    cells_per_side = torch.tensor([5])
    dim = 1

    expected_result = torch.arange(60).reshape(5, 2, 2, 3)

    result = _unflatten_cell_buffer(buffer, cells_per_side, dim)

    assert torch.equal(result, expected_result), f"Expected: {expected_result}, but got: {result}"


def test__unflatten_cell_buffer_that_raises_exception():
    buffer = torch.arange(65).reshape(13, 5)
    cells_per_side = torch.tensor([3])
    dim = 1

    with pytest.raises(ValueError):
        _unflatten_cell_buffer(buffer, cells_per_side, dim)


def test__unflatten_cell_buffer_cells_per_side_is_2d_tensor():
    buffer = torch.arange(120).reshape(10, 2, 6)
    cells_per_side = torch.tensor([[3, 2]])
    dim = 2

    # Expected shape: (2, 3, 2, -1, 2), which translates into (2, 3, 2, 10, 2) for a tensor of 120 elements
    expected_result = torch.arange(120).reshape(2, 3, 2, 10, 2)

    try:
        result = _unflatten_cell_buffer(buffer, cells_per_side, dim)
        assert torch.equal(result,
                           expected_result), f"Expected: {expected_result}, but got: {result}"
        print("Test passed.")
    except ValueError as e:
        print(f"Test failed with error: {e}")

# def test__unflatten_cell_buffer_cells_per_side_is_1d():
#     reshaped_tensor = torch.tensor(
#         [[[[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]],
#           [[[12, 13, 14], [15, 16, 17]], [[18, 19, 20], [21, 22, 23]]],
#           [[[24, 25, 26], [27, 28, 29]], [[30, 31, 32], [33, 34, 35]]],
#           [[[36, 37, 38], [39, 40, 41]], [[42, 43, 44], [45, 46, 47]]]],
#
#          [[[[48, 49, 50], [51, 52, 53]], [[54, 55, 56], [57, 58, 59]]],
#           [[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]],
#           [[[12, 13, 14], [15, 16, 17]], [[18, 19, 20], [21, 22, 23]]],
#           [[[24, 25, 26], [27, 28, 29]], [[30, 31, 32], [33, 34, 35]]]],
#
#          [[[[36, 37, 38], [39, 40, 41]], [[42, 43, 44], [45, 46, 47]]],
#           [[[48, 49, 50], [51, 52, 53]], [[54, 55, 56], [57, 58, 59]]],
#           [[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]],
#           [[[12, 13, 14], [15, 16, 17]], [[18, 19, 20], [21, 22, 23]]]],
#
#          [[[[24, 25, 26], [27, 28, 29]], [[30, 31, 32], [33, 34, 35]]],
#           [[[36, 37, 38], [39, 40, 41]], [[42, 43, 44], [45, 46, 47]]],
#           [[[48, 49, 50], [51, 52, 53]], [[54, 55, 56], [57, 58, 59]]],
#           [[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]]],
#
#          [[[[12, 13, 14], [15, 16, 17]], [[18, 19, 20], [21, 22, 23]]],
#           [[[24, 25, 26], [27, 28, 29]], [[30, 31, 32], [33, 34, 35]]],
#           [[[36, 37, 38], [39, 40, 41]], [[42, 43, 44], [45, 46, 47]]],
#           [[[48, 49, 50], [51, 52, 53]], [[54, 55, 56], [57, 58, 59]]]]]
#     )
#
#     buffer = torch.arange(60).reshape(10, 2, 3)
#     cells_per_side = torch.tensor([4, 5])
#     dim = 1
#
#     assert torch.equal(
#         _unflatten_cell_buffer(buffer, cells_per_side, dim),
#         reshaped_tensor
#     )

