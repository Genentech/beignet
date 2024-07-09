# import functools
# from typing import Callable
#
# import beignet.func
# import hypothesis
# import hypothesis.strategies
# import torch.testing
# from torch import Tensor
#
#
# def map_product(fn: Callable) -> Callable:
#     return torch.vmap(
#         torch.vmap(
#             fn,
#             in_dims=(0, None),
#             out_dims=0,
#         ),
#         in_dims=(None, 0),
#         out_dims=0,
#     )
#
#
# @hypothesis.strategies.composite
# def _strategy(function):
#     dtype = function(
#         hypothesis.strategies.sampled_from(
#             [
#                 torch.float32,
#                 torch.float64,
#             ],
#         ),
#     )
#
#     maximum_size = function(
#         hypothesis.strategies.floats(
#             min_value=1.0,
#             max_value=8.0,
#         ),
#     )
#
#     particles = function(
#         hypothesis.strategies.integers(
#             min_value=16,
#             max_value=32,
#         ),
#     )
#
#     spatial_dimension = function(
#         hypothesis.strategies.integers(
#             min_value=1,
#             max_value=3,
#         ),
#     )
#
#     return (
#         dtype,
#         torch.rand([particles, spatial_dimension], dtype=dtype),
#         particles,
#         torch.rand([spatial_dimension], dtype=dtype) * maximum_size,
#         spatial_dimension,
#     )
#
#
# @hypothesis.given(_strategy())
# @hypothesis.settings(deadline=None)
# def test_space(data):
#     dtype, input, particles, size, spatial_dimension = data
#
#     displacement_fn, shift_fn = beignet.func.space(size, parallelepiped=False)
#
#     (
#         parallelepiped_displacement_fn,
#         parallelepiped_shift_fn,
#     ) = beignet.func.space(
#         torch.diag(size),
#     )
#
#     standardized_input = input * size
#
#     displacement_fn = map_product(displacement_fn)
#
#     parallelepiped_displacement_fn = map_product(
#         parallelepiped_displacement_fn,
#     )
#
#     torch.testing.assert_close(
#         displacement_fn(
#             standardized_input,
#             standardized_input,
#         ),
#         parallelepiped_displacement_fn(
#             input,
#             input,
#         ),
#     )
#
#     displacement = torch.randn([particles, spatial_dimension], dtype=dtype)
#
#     torch.testing.assert_close(
#         shift_fn(standardized_input, displacement),
#         parallelepiped_shift_fn(input, displacement) * size,
#     )
#
#     def f(input: Tensor) -> Tensor:
#         return torch.sum(displacement_fn(input, input) ** 2)
#
#     def g(input: Tensor) -> Tensor:
#         return torch.sum(parallelepiped_displacement_fn(input, input) ** 2)
#
#     torch.testing.assert_close(
#         torch.func.grad(f)(standardized_input),
#         torch.func.grad(g)(input),
#         rtol=0.0001,
#         atol=0.0001,
#     )
#
#     size_a = 10.0 * torch.rand([])
#     size_b = 10.0 * torch.rand([], dtype=dtype)
#
#     transform_a = 0.5 * torch.randn([spatial_dimension, spatial_dimension])
#     transform_b = 0.5 * torch.randn(
#         [spatial_dimension, spatial_dimension],
#         dtype=dtype,
#     )
#
#     transform_a = size_a * (torch.eye(spatial_dimension) + transform_a)
#     transform_b = size_b * (torch.eye(spatial_dimension) + transform_b)
#
#     displacement_fn_a, shift_fn_a = beignet.func.space(transform_a)
#     displacement_fn_b, shift_fn_b = beignet.func.space(transform_b)
#
#     displacement = torch.randn([particles, spatial_dimension], dtype=dtype)
#
#     torch.testing.assert_close(
#         map_product(
#             functools.partial(
#                 displacement_fn_a,
#                 transform=transform_b,
#             ),
#         )(input, input),
#         map_product(displacement_fn_b)(input, input),
#     )
#
#     torch.testing.assert_close(
#         shift_fn_a(input, displacement, transform=transform_b),
#         shift_fn_b(input, displacement),
#     )
