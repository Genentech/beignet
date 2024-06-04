# import pytest
# import torch
#
# from beignet.func._molecular_dynamics._partition.__unflatten_cell_buffer import \
#     _unflatten_cell_buffer
#
#
# def test_unflatten_cell_buffer():
#     # Test with 1D tensor cells_per_side
#     buffer = torch.arange(24)
#     cells_per_side = torch.tensor([4, 3])
#     reshaped = _unflatten_cell_buffer(buffer, cells_per_side, 2)
#     assert reshaped.shape == (3, 4, 2)
#
#     # Test with scalar int cells_per_side
#     buffer = torch.arange(24)
#     cells_per_side = 3
#     reshaped = _unflatten_cell_buffer(buffer, cells_per_side, 2)
#     assert reshaped.shape == (3, 3, 2, 1)
#
#     # Test with scalar float cells_per_side
#     buffer = torch.arange(24)
#     cells_per_side = 2.0
#     reshaped = _unflatten_cell_buffer(buffer, cells_per_side, 2)
#     assert reshaped.shape == (2, 2, 6, 1)
#
#     # Test with 2D tensor cells_per_side
#     buffer = torch.arange(24)
#     cells_per_side = torch.tensor([[4, 3]])
#     reshaped = _unflatten_cell_buffer(buffer, cells_per_side, 2)
#     assert reshaped.shape == (3, 4, 2)
#
#
# def test_invalid_cells_per_side():
#     buffer = torch.arange(24)
#     cells_per_side = torch.tensor([4, 3, 2])
#     with pytest.raises(ValueError):
#         _unflatten_cell_buffer(buffer, cells_per_side, 2)
#
#
# def test_incompatible_buffer_size():
#     buffer = torch.arange(25)
#     cells_per_side = 3
#     with pytest.raises(ValueError):
#         _unflatten_cell_buffer(buffer, cells_per_side, 2)