# import torch
# import hypothesis.strategies as st
# from hypothesis import given
# import pytest
#
# from beignet.func._molecular_dynamics._partition.__particles_per_cell import \
#     _particles_per_cell
#
#
# @st.composite
# def _particles_per_cell_strategy(draw):
#     num_particles = draw(st.integers(min_value=1, max_value=100))
#     spatial_dimensions = draw(st.integers(min_value=1, max_value=5))
#
#     positions = draw(
#         st.lists(
#             st.lists(
#                 st.floats(min_value=0.0, max_value=10.0),
#                 min_size=3, max_size=spatial_dimensions
#             ),
#             min_size=num_particles, max_size=num_particles
#         ).map(torch.tensor)
#     )
#
#     size = draw(
#         st.lists(
#             st.floats(min_value=1.0, max_value=10.0),
#             min_size=spatial_dimensions, max_size=spatial_dimensions
#         ).map(torch.tensor)
#     )
#
#     minimum_size = draw(st.floats(min_value=0.1, max_value=10.0))
#
#     return positions, size, minimum_size
#
#
# @pytest.mark.parametrize(
#     "positions, size, minimum_size, expected_exception",
#     [
#         (torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([5.0]), 1.0, AssertionError),  # Size not matching spatial dimensions
#         (torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]), torch.tensor([5.0, 5.0, 5.0]), 1.0, ValueError),  # Positions shape not matching (n, d)
#     ]
# )
# def test_particles_per_cell_exceptions(positions, size, minimum_size, expected_exception):
#     """
#     Test the `_particles_per_cell` function for expected exceptions based on input parameters.
#     """
#     if expected_exception is not None:
#         with pytest.raises(expected_exception):
#             _particles_per_cell(positions, size, minimum_size)
#     else:
#         _particles_per_cell(positions, size, minimum_size)
#
#
# @given(_particles_per_cell_strategy())
# def test__particles_per_cell(data):
#     """
#     Property-based test for the `_particles_per_cell` function ensuring correct behavior for various inputs.
#     """
#     positions, size, minimum_size = data
#
#     # Validate that dimensions of size match the spatial dimensions of positions.
#     if size.numel() != positions.shape[1]:
#         with pytest.raises(ValueError):
#             _particles_per_cell(positions, size, minimum_size)
#         return
#
#     result = _particles_per_cell(positions, size, minimum_size)
#
#     # Validate the shape and type of the result.
#     assert result.dtype == torch.int32  # Assuming the counts will be integers.
#     assert result.numel() == len(result)  # Ensure the result is a 1D tensor.
#
#     # Validate that the sum of the result is equal to the number of particles.
#     assert torch.sum(result).item() == positions.shape[0]
