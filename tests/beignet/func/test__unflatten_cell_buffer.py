# import hypothesis.strategies as st
# import math
# import pytest
# import torch
# from hypothesis import given
# from torch import Tensor
#
# from beignet.func._molecular_dynamics._partition.__unflatten_cell_buffer import \
#     _unflatten_cell_buffer
#
#
# @st.composite
# def _unflatten_cell_buffer_strategy(draw):
#     # Generate the dimension value
#     dim = draw(st.integers(min_value=1, max_value=3))
#
#     # Generate cells_per_side options
#     cells_per_side_scalar = draw(st.integers(min_value=1, max_value=10))
#     cells_per_side_1d = torch.tensor(draw(
#         st.lists(st.integers(min_value=1, max_value=10), min_size=dim,
#                  max_size=dim)))
#     cells_per_side_2d = torch.tensor(draw(st.lists(
#         st.lists(st.integers(min_value=1, max_value=10), min_size=1,
#                  max_size=1), min_size=dim, max_size=dim)))
#
#     cells_options = [cells_per_side_scalar, cells_per_side_1d,
#                      cells_per_side_2d]
#     cells_per_side = draw(st.sampled_from(cells_options))
#
#     if isinstance(cells_per_side, (int, float)):
#         num_elements = cells_per_side ** dim
#     elif isinstance(cells_per_side, Tensor) and len(cells_per_side.shape) == 1:
#         num_elements = torch.prod(cells_per_side).item()
#     elif isinstance(cells_per_side, Tensor) and len(cells_per_side.shape) == 2:
#         num_elements = torch.prod(cells_per_side[0]).item()
#
#     if num_elements == 0:  # Ensure num_elements is non-zero
#         num_elements = 1
#
#     buffer = draw(
#         st.lists(
#             st.floats(min_value=-1.0, max_value=1.0),
#             min_size=num_elements,
#             max_size=num_elements
#         ).map(torch.tensor).map(lambda x: x.view([num_elements]))
#     )
#
#     return buffer, cells_per_side, dim
#
#
# @pytest.mark.parametrize(
#     "buffer, cells_per_side, dim, expected_exception",
#     [
#         (torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([1, 2, 3, 4]), 2, ValueError),  # cells_per_side has too many elements
#         (torch.tensor([1.0, 2.0]), torch.tensor([1.0]), 2, ValueError),  # Invalid shape for cells_per_side
#     ]
# )
# def test_unflatten_cell_buffer_exceptions(buffer, cells_per_side, dim, expected_exception):
#     """
#     Test the `_unflatten_cell_buffer` function for expected exceptions based on input parameters.
#     """
#     if expected_exception is not None:
#         with pytest.raises(expected_exception):
#             _unflatten_cell_buffer(buffer, cells_per_side, dim)
#     else:
#         _unflatten_cell_buffer(buffer, cells_per_side, dim)
#
#
# @given(_unflatten_cell_buffer_strategy())
# def test__unflatten_cell_buffer(data):
#     """
#     Property-based test for the `_unflatten_cell_buffer` function ensuring correct behavior for various inputs.
#     """
#     buffer, cells_per_side, dim = data
#
#     result = _unflatten_cell_buffer(buffer, cells_per_side, dim)
#
#     # Validate the shape
#     if isinstance(cells_per_side, (int, float)):
#         unflatten_shape = (int(cells_per_side),) * dim + (-1,) + buffer.shape[1:]
#     elif isinstance(cells_per_side, Tensor) and len(cells_per_side.shape) == 1:
#         unflatten_shape = tuple(cells_per_side.flip(0)) + (-1,) + buffer.shape[1:]
#     elif isinstance(cells_per_side, Tensor) and len(cells_per_side.shape) == 2:
#         unflatten_shape = tuple(cells_per_side[0].flip(0)) + (-1,) + buffer.shape[1:]
#
#     assert result.shape == torch.Size(unflatten_shape)
#
#     # Validate the content
#     expected_result = torch.reshape(buffer, result.shape)
#     assert torch.equal(result, expected_result)
