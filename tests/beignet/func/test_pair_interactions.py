import dataclasses

import pytest
import torch
from torch import Tensor

from beignet.func import space
from beignet.func._interact import interact, _ParameterTreeKind, _ParameterTree
from beignet.func._partition import distance, metric, neighbor_list, \
    _NeighborListFormat

PARTICLE_COUNT = 1000
NEIGHBOR_LIST_PARTICLE_COUNT = 100
STOCHASTIC_SAMPLES = 3
POSITION_DTYPE = [torch.float32, torch.float64]  # Example values
SPATIAL_DIMENSION = [2, 3]
NEIGHBOR_LIST_FORMAT = [
    _NeighborListFormat.DENSE,
    _NeighborListFormat.ORDERED_SPARSE,
    _NeighborListFormat.SPARSE
]
test_cases = [
    {
        "dtype": dtype,
        "dim": dim,
    }
    for dtype in POSITION_DTYPE
    for dim in SPATIAL_DIMENSION
]

dtype_fmt_test_cases = [
    {
        "dtype": dtype,
        "fmt": fmt,
    }
    for dtype in POSITION_DTYPE
    for fmt in NEIGHBOR_LIST_FORMAT
]

neighbor_list_test_cases = [
    {
        "dtype": dtype,
        "dim": dim,
        "fmt": fmt,
    }
    for dtype in POSITION_DTYPE
    for dim in SPATIAL_DIMENSION
    for fmt in NEIGHBOR_LIST_FORMAT
]

# Extract the parameters and ids
params = [(case["dtype"], case["dim"]) for case in test_cases]
dtype_fmt_params = [(case["dtype"], case["fmt"]) for case in dtype_fmt_test_cases]
neighbor_list_params = [(case["dtype"], case["dim"], case["fmt"]) for case in neighbor_list_test_cases]


# @pytest.mark.parametrize("dtype, dim", params)
# def test_pair_no_kinds_scalar(dtype, dim):
#     square = lambda dr: dr ** 2
#
#     displacement, _ = space(box=None)
#
#     metric = lambda Ra, Rb, **kwargs: \
#         torch.sum(displacement(Ra, Rb, **kwargs) ** 2, dim=-1)
#
#     mapped_square = interact(
#         fn=square,
#         displacement_fn=metric,
#         interaction="pair",
#     )
#
#     metric = map_product(metric)
#
#     torch.manual_seed(0)
#
#     for _ in range(STOCHASTIC_SAMPLES):
#         R = torch.rand((PARTICLE_COUNT, dim), dtype=dtype)
#
#         assert torch.allclose(
#             mapped_square(R),
#             torch.tensor(0.5 * torch.sum(square(metric(R, R))), dtype=dtype)
#         )
#
#
# @pytest.mark.parametrize("dtype", POSITION_DTYPE)
# def test_pair_no_kinds_pytree(dtype):
#     square_scalar = lambda dr, p0, p1: p0 * dr ** 2 + p1
#     square_higher = lambda dr, p: p[0] * dr ** 2 + p[1]
#
#     @dataclasses.dataclass
#     class Parameter:
#         scale: Tensor
#         shift: Tensor
#
#     tree_fn = lambda dr, p: p.scale * dr ** 2 + p.shift
#     displacement, _ = space(box=None)
#     test_metric = metric(displacement)
#
#     p = torch.tensor([1.0, 2.0])
#     M = _ParameterTreeKind
#     mapped_scalar = interact(
#         square_scalar,
#         test_metric,
#         interaction="pair",
#         p0=p[0],
#         p1=p[1],
#     )
#     mapped_higher = interact(
#         square_higher,
#         test_metric,
#         interaction="pair",
#         p=_ParameterTree(p, M.SPACE)
#     )
#
#     p_tree = _ParameterTree(Parameter(scale=p[0], shift=p[1]), M.SPACE)
#     mapped_tree = interact(tree_fn, test_metric, p=p_tree, interaction="pair")
#
#     torch.manual_seed(0)
#
#     for _ in range(STOCHASTIC_SAMPLES):
#         R = torch.rand((PARTICLE_COUNT, 2), dtype=dtype)
#
#         assert torch.allclose(mapped_scalar(R), mapped_higher(R))
#         assert torch.allclose(mapped_scalar(R), mapped_tree(R))
#
#
# @pytest.mark.parametrize("dtype, dim", params)
# def test_pair_no_kinds_scalar_dynamic(dtype, dim):
#     square = lambda dr, epsilon: epsilon * dr ** 2
#     displacement, _ = space(box=None)
#     metric = lambda Ra, Rb, **kwargs: \
#         torch.sum(displacement(Ra, Rb, **kwargs) ** 2, dim=-1)
#
#     mapped_square = interact(square, metric, epsilon=1.0, interaction="pair")
#     metric = map_product(metric)
#
#     torch.manual_seed(0)
#
#     for _ in range(STOCHASTIC_SAMPLES):
#         R = torch.rand((PARTICLE_COUNT, dim), dtype=dtype)
#         epsilon = torch.rand((PARTICLE_COUNT,), dtype=dtype)
#         mat_epsilon = 0.5 * (epsilon[:, None] + epsilon[None, :])
#
#         assert torch.allclose(
#             mapped_square(R, epsilon=epsilon),
#             torch.tensor(
#                 0.5 * torch.sum(square(metric(R, R), mat_epsilon)), dtype=dtype
#             )
#         )
#
#
# @pytest.mark.parametrize("dtype, dim", params)
# def test_pair_no_kinds_vector(dtype, dim):
#     square = lambda dr: torch.sum(dr ** 2, dim=2)
#     displacement, _ = space(box=None)
#
#     mapped_square = interact(square, displacement, interaction="pair")
#
#     displacement = map_product(displacement)
#     torch.manual_seed(0)
#
#     for _ in range(STOCHASTIC_SAMPLES):
#         R = torch.rand((PARTICLE_COUNT, dim), dtype=dtype)
#         mapped_ref = torch.tensor(0.5 * torch.sum(square(displacement(R, R))),
#                                   dtype=dtype)
#
#         assert torch.allclose(mapped_square(R), mapped_ref)
#
#
# # TODO (isaacsoh) broken on line 264 of _interacte (10 clicks)
# @pytest.mark.parametrize("dtype", POSITION_DTYPE)
# def test_pair_no_kinds_pytree_per_particle(dtype):
#     square_scalar = lambda dr, p0, p1: p0 * dr ** 2 + p1
#     square_higher = lambda dr, p: p[..., 0] * dr ** 2 + p[..., 1]
#
#     @dataclasses.dataclass
#     class Parameter:
#         scale: Tensor
#         shift: Tensor
#
#     tree_fn = lambda dr, p: p.scale * dr**2 + p.shift
#
#     displacement, _ = space(box=None)
#     my_metric = metric(displacement)
#
#     p = torch.rand((PARTICLE_COUNT, 2))
#     M = _ParameterTreeKind
#     mapped_scalar = interact(square_scalar, my_metric, p0=p[:, 0], p1=p[:, 1], interaction="pair")
#     p_higher = _ParameterTree(p, M.PARTICLE)
#     mapped_higher = interact(square_higher, my_metric, p=p_higher, interaction="pair")
#
#     p_tree = _ParameterTree(Parameter(scale=p[:, 0], shift=p[:, 1]), M.PARTICLE)
#     mapped_tree = interact(tree_fn, my_metric, p=p_tree, interaction="pair")
#
#     torch.manual_seed(0)
#
#     R = torch.rand((PARTICLE_COUNT, 2), dtype=dtype)
#     assert torch.allclose(mapped_scalar(R), mapped_higher(R))
#     # assert torch.allclose(mapped_scalar(R), mapped_tree(R))
#
#
# @pytest.mark.parametrize("dtype", POSITION_DTYPE)
# def test_pair_no_kinds_pytree_order_per_bond(dtype):
#     square_scalar = lambda dr, p0, p1: p0 * dr ** 2 + p1
#     square_higher = lambda dr, p: p[..., 0] * dr ** 2 + p[..., 1]
#
#     @dataclasses.dataclass
#     class Parameter:
#         scale: Tensor
#         shift: Tensor
#
#     tree_fn = lambda dr, p: p.scale * dr**2 + p.shift
#
#     displacement, _ = space(box=None)
#     my_metric = metric(displacement)
#
#     p = torch.rand((PARTICLE_COUNT, PARTICLE_COUNT, 2))
#     M = _ParameterTreeKind
#
#     mapped_scalar = interact(square_scalar, my_metric, p0=p[..., 0], p1=p[..., 1], interaction="pair")
#     mapped_higher = interact(square_higher, my_metric, p=_ParameterTree(p, M.BOND), interaction="pair")
#
#     p_tree = _ParameterTree(Parameter(scale=p[..., 0], shift=p[..., 1]), M.BOND)
#
#     mapped_tree = interact(tree_fn, my_metric, p=p_tree, interaction="pair")
#
#     torch.manual_seed(0)
#
#     for _ in range(STOCHASTIC_SAMPLES):
#         R = torch.rand((PARTICLE_COUNT, 2), dtype=dtype)
#         assert torch.allclose(mapped_scalar(R), mapped_higher(R))
#         assert torch.allclose(mapped_scalar(R), mapped_tree(R))
#
#
# @pytest.mark.parametrize("dtype, dim", params)
# def test_pair_no_kinds_vector_nonadditive(dtype, dim):
#     square = lambda dr, params: params * torch.sum(dr ** 2, dim=2)
#     disp, _ = space(box=None)
#
#     mapped_square = interact(square, disp, params=lambda x, y: x * y,
#                              interaction="pair")
#
#     disp = map_product(disp)
#     torch.manual_seed(0)
#
#     for _ in range(STOCHASTIC_SAMPLES):
#         R = torch.rand((PARTICLE_COUNT, dim), dtype=dtype)
#         params = torch.rand((PARTICLE_COUNT,),
#                             dtype=dtype) * 1.4 + 0.1  # minval=0.1, maxval=1.5
#         pp_params = params[None, :] * params[:, None]
#         mapped_ref = torch.tensor(
#             0.5 * torch.sum(square(disp(R, R), pp_params)), dtype=dtype)
#
#         assert torch.allclose(mapped_square(R, params=params), mapped_ref)
#
#
# @pytest.mark.parametrize("dtype, dim", params)
# def test_pair_static_kinds_scalar(dtype, dim):
#     torch.manual_seed(0)
#
#     square = lambda dr, param=1.0: param * dr ** 2
#     params = torch.tensor([[1.0, 2.0], [2.0, 3.0]], dtype=torch.float32)
#
#     kinds = torch.randint(0, 2, (PARTICLE_COUNT,))
#
#     displacement, _ = space(box=None)
#     metric = lambda Ra, Rb, **kwargs: torch.sum(displacement(Ra, Rb, **kwargs) ** 2, dim=-1)
#
#     mapped_square = interact(square, metric, kinds=kinds, param=params, interaction="pair")
#
#     metric = map_product(metric)
#
#     for _ in range(STOCHASTIC_SAMPLES):
#         R = torch.rand((PARTICLE_COUNT, dim), dtype=dtype)
#         total = 0.0
#         for i in range(2):
#             for j in range(2):
#                 param = params[i, j]
#                 R_1 = R[kinds == i]
#                 R_2 = R[kinds == j]
#                 total += 0.5 * torch.sum(square(metric(R_1, R_2), param))
#         assert torch.allclose(mapped_square(R), torch.tensor(total, dtype=dtype))
#
#
# @pytest.mark.parametrize("dtype, dim", params)
# def test_pair_static_kinds_scalar_dynamic(dtype, dim):
#     torch.manual_seed(0)
#
#     square = lambda dr, param=1.0: param * dr ** 2
#
#     kinds = torch.randint(0, 2, (PARTICLE_COUNT,))
#
#     displacement, _ = space(box=None)
#     metric = lambda Ra, Rb, **kwargs: torch.sum(displacement(Ra, Rb, **kwargs) ** 2, dim=-1)
#
#     mapped_square = interact(square, metric, kinds=kinds, param=1.0, interaction="pair")
#
#     metric = map_product(metric)
#
#     for _ in range(STOCHASTIC_SAMPLES):
#         R = torch.rand((PARTICLE_COUNT, dim), dtype=dtype)
#         params = torch.rand((2, 2), dtype=dtype)
#         params = 0.5 * (params + params.T)
#         total = 0.0
#         for i in range(2):
#             for j in range(2):
#                 param = params[i, j]
#                 R_1 = R[kinds == i]
#                 R_2 = R[kinds == j]
#                 total += 0.5 * torch.sum(square(metric(R_1, R_2), param))
#         assert torch.allclose(mapped_square(R, param=params), torch.tensor(total, dtype=dtype))
#
#
# @pytest.mark.parametrize("dtype, dim", params)
# def test_pair_scalar_dummy_arg(dtype, dim):
#     torch.manual_seed(0)
#
#     square = lambda dr, param=torch.tensor(1.0, dtype=torch.float32), **unused_kwargs: param * dr ** 2
#
#     R = torch.randn((PARTICLE_COUNT, dim), dtype=dtype)
#     displacement, shift = space(box=None)
#
#     mapped = interact(square, metric(displacement), interaction="pair")
#
#     mapped(R, t=torch.tensor(0.0, dtype=torch.float32))
#
#
# @pytest.mark.parametrize("dtype", POSITION_DTYPE)
# def test_pair_kinds_pytree_global(dtype):
#     square_scalar = lambda dr, p0, p1: p0 * dr ** 2 + p1
#     square_higher = lambda dr, p: p[..., 0] * dr ** 2 + p[..., 1]
#
#     @dataclasses.dataclass
#     class Parameter:
#         scale: Tensor
#         shift: Tensor
#
#     square_tree = lambda dr, p: p.scale * dr ** 2 + p.shift
#
#     displacement, _ = space(box=None)
#     my_metric = metric(displacement)
#
#     p = torch.tensor([1.0, 2.0])
#     M = _ParameterTreeKind
#     kinds = torch.where(torch.arange(PARTICLE_COUNT) < PARTICLE_COUNT // 2,
#                           0, 1)
#
#     mapped_scalar = interact(square_scalar, my_metric, kinds=kinds,
#                              p0=p[0], p1=p[1], interaction="pair")
#     p_h = _ParameterTree(p, M.SPACE)
#     mapped_higher = interact(square_higher, my_metric, kinds=kinds, p=p_h,
#                              interaction="pair")
#
#     p_tree = _ParameterTree(Parameter(p[0], p[1]), M.SPACE)
#     mapped_tree = interact(square_tree, my_metric, kinds=kinds, p=p_tree,
#                            interaction="pair")
#
#     torch.manual_seed(0)
#
#     for _ in range(STOCHASTIC_SAMPLES):
#         R = torch.rand((PARTICLE_COUNT, 2), dtype=dtype)
#         assert torch.allclose(mapped_scalar(R), mapped_higher(R))
#         assert torch.allclose(mapped_scalar(R), mapped_tree(R))
# #
# #
# # TODO (isaacsoh) broken on line 264 of _interacte (10 clicks)
# @pytest.mark.parametrize("dtype", POSITION_DTYPE)
# def test_pair_kinds_pytree_per_kinds(dtype):
#     square_scalar = lambda dr, p0, p1: p0 * dr ** 2 + p1
#     square_higher = lambda dr, p: p[..., 0] * dr ** 2 + p[..., 1]
#
#     @dataclasses.dataclass
#     class Parameter:
#         scale: torch.Tensor
#         shift: torch.Tensor
#
#     square_tree = lambda dr, p: p.scale * dr ** 2 + p.shift
#
#     displacement, _ = space(box=None)
#     my_metric = metric(displacement)
#
#     p = torch.rand((2, 2, 2))
#     p = p + p.transpose(0, 1)
#     kinds = torch.where(torch.arange(PARTICLE_COUNT) < PARTICLE_COUNT // 2,
#                           0, 1)
#
#     mapped_scalar = interact(square_scalar, my_metric, kinds=kinds,
#                              p0=p[..., 0], p1=p[..., 1], interaction="pair")
#     M = _ParameterTreeKind
#     p_h = _ParameterTree(p, M.KINDS)
#     mapped_higher = interact(square_higher, my_metric, kinds=kinds, p=p_h,
#                              interaction="pair")
#
#     p_tree = _ParameterTree(Parameter(p[..., 0], p[..., 1]), M.KINDS)
#     mapped_tree = interact(square_tree, my_metric, kinds=kinds, p=p_tree,
#                            interaction="pair")
#
#     torch.manual_seed(0)
#
#     for _ in range(STOCHASTIC_SAMPLES):
#         R = torch.rand((PARTICLE_COUNT, 2), dtype=dtype)
#         assert torch.allclose(mapped_scalar(R), mapped_higher(R))
#         # assert torch.allclose(mapped_scalar(R), mapped_tree(R))
#
#
# @pytest.mark.parametrize("dtype, dim", params)
# def test_pair_static_kinds_vector(dtype, dim):
#     torch.manual_seed(0)
#
#     square = lambda dr, param=1.0: param * torch.sum(dr ** 2, dim=2)
#     params = torch.tensor([[1.0, 2.0], [2.0, 3.0]], dtype=torch.float32)
#
#     kinds = torch.randint(0, 2, (PARTICLE_COUNT,))
#
#     displacement, _ = space(box=None)
#
#     mapped_square = interact(square, displacement, kinds=kinds, param=params, interaction="pair")
#
#     displacement = map_product(displacement)
#
#     for _ in range(STOCHASTIC_SAMPLES):
#         R = torch.rand((PARTICLE_COUNT, dim), dtype=dtype)
#         total = 0.0
#         for i in range(2):
#             for j in range(2):
#                 param = params[i, j]
#                 R_1 = R[kinds == i]
#                 R_2 = R[kinds == j]
#                 total += 0.5 * torch.sum(square(displacement(R_1, R_2), param))
#         assert torch.allclose(mapped_square(R), torch.tensor(total, dtype=dtype))
#
#
# @pytest.mark.parametrize("dtype, dim", params)
# def test_pair_dynamic_kinds_scalar(dtype, dim):
#     torch.manual_seed(0)
#
#     square = lambda dr, param=1.0: param * dr ** 2
#     params = torch.tensor([[1.0, 2.0], [2.0, 3.0]], dtype=torch.float32)
#
#     kinds = torch.randint(0, 2, (PARTICLE_COUNT,))
#
#     displacement, _ = space(box=None)
#     my_metric = metric(displacement)
#
#     mapped_square = interact(square, my_metric, kinds=2, param=params, interaction="pair")
#
#     my_metric = map_product(my_metric)
#
#     for _ in range(STOCHASTIC_SAMPLES):
#         R = torch.rand((PARTICLE_COUNT, dim), dtype=dtype)
#         total = 0.0
#         for i in range(2):
#             for j in range(2):
#                 param = params[i, j]
#                 R_1 = R[kinds == i]
#                 R_2 = R[kinds == j]
#                 total += 0.5 * torch.sum(square(my_metric(R_1, R_2), param))
#         assert torch.allclose(mapped_square(R, kinds), torch.tensor(total, dtype=dtype))
#
#
# @pytest.mark.parametrize("dtype, dim", params)
# def test_pair_dynamic_kinds_vector(dtype, dim):
#     torch.manual_seed(0)
#
#     square = lambda dr, param=1.0: param * torch.sum(dr ** 2, dim=-1)
#     params = torch.tensor([[1.0, 2.0], [2.0, 3.0]], dtype=torch.float32)
#
#     kinds = torch.randint(0, 2, (PARTICLE_COUNT,))
#
#     displacement, _ = space(box=None)
#
#     mapped_square = interact(square, displacement, kinds=2, param=params, interaction="pair")
#
#     disp = torch.vmap(torch.vmap(displacement, (0, None), 0), (None, 0), 0)
#
#     for _ in range(STOCHASTIC_SAMPLES):
#         R = torch.rand((PARTICLE_COUNT, dim), dtype=dtype)
#         total = 0.0
#         for i in range(2):
#             for j in range(2):
#                 param = params[i, j]
#                 R_1 = R[kinds == i]
#                 R_2 = R[kinds == j]
#                 total += 0.5 * torch.sum(square(disp(R_1, R_2), param))
#         assert torch.allclose(mapped_square(R, kinds), torch.tensor(total, dtype=dtype))
#
#
# @pytest.mark.parametrize("dtype, dim, fmt", neighbor_list_params)
# def test_pair_neighbor_list_scalar(dtype, dim, fmt):
#     torch.manual_seed(0)
#
#     def truncated_square(dr: Tensor, sigma: Tensor) -> Tensor:
#         return torch.where(dr < sigma, dr ** 2, torch.tensor(0.0, dtype=torch.float32))
#
#     N = NEIGHBOR_LIST_PARTICLE_COUNT
#     box_size = 4.0 * N ** (1.0 / dim)
#
#     displacement, _ = space(box=box_size, parallelepiped=False)
#     d = metric(displacement)
#
#     neighbor_square = interact(truncated_square, d, sigma=1.0, interaction="neighbor_list")
#     mapped_square = interact(truncated_square, d, sigma=1.0, interaction="pair")
#
#     for _ in range(STOCHASTIC_SAMPLES):
#         R = box_size * torch.rand((N, dim), dtype=dtype)
#         sigma = torch.rand(()) * 2.0 + 0.5  # minval=0.5, maxval=2.5
#         neighbor_fn = neighbor_list(displacement, torch.tensor([box_size] * dim, dtype=dtype), sigma, 0.0, neighbor_list_format=fmt)
#         nbrs = neighbor_fn.setup_fn(R)
#         assert torch.allclose(mapped_square(R, sigma=sigma), neighbor_square(R, nbrs, sigma=sigma))
#


# @pytest.mark.parametrize("dtype, fmt", dtype_fmt_params)
# def test_pair_neighbor_list_pytree(dtype, fmt):
#     torch.manual_seed(0)
#     dim = 2
#
#     def scalar_fn(dr: Tensor, sigma: Tensor, shift: Tensor) -> Tensor:
#         return torch.where(dr < sigma, dr ** 2 + shift, torch.tensor(0.0, dtype=torch.float32))
#
#     def higher_order_fn(dr: Tensor, p: Tensor) -> Tensor:
#         sigma = torch.rand(()) * 2.0 + 0.5  # minval=0.5, maxval=2.5
#         return torch.where(dr < p[..., 0], dr ** 2 + p[..., 1], torch.tensor(0.0, dtype=torch.float32))
#
#     @dataclasses.dataclass
#     class Parameter:
#         sigma: torch.Tensor
#         shift: torch.Tensor
#
#     def tree_fn(dr: Tensor, p):
#         return torch.where(dr < p.sigma, dr ** 2 + p.shift, torch.tensor(0.0, dtype=torch.float32))
#
#     N = NEIGHBOR_LIST_PARTICLE_COUNT
#     box_size = 4.0 * N ** (1.0 / dim)
#
#     displacement, _ = space(box=box_size, parallelepiped=False)
#     d = metric(displacement)
#
#     sigma = torch.tensor(1.0, dtype=torch.float32)
#     shift = torch.tensor(2.0, dtype=torch.float32)
#     M = _ParameterTreeKind
#
#     neighbor_scalar = interact(scalar_fn, d, sigma=sigma, shift=shift, interaction="neighbor_list")
#     p = _ParameterTree(torch.tensor([sigma, shift], dtype=dtype), M.SPACE)
#     neighbor_higher = interact(higher_order_fn, d, p=p, interaction="neighbor_list")
#
#     p_tree = _ParameterTree(Parameter(sigma=sigma, shift=shift), M.SPACE)
#     neighbor_tree = interact(tree_fn, d, p=p_tree, interaction="neighbor_list")
#
#     for _ in range(STOCHASTIC_SAMPLES):
#         R = box_size * torch.rand((N, dim), dtype=dtype)
#         sigma = torch.rand(()) * 2.0 + 0.5  # minval=0.5, maxval=2.5
#         neighbor_fn = neighbor_list(displacement, torch.tensor([box_size] * dim, dtype=dtype), sigma, 0.0, neighbor_list_format=fmt)
#         nbrs = neighbor_fn.setup_fn(R)
#         assert torch.allclose(neighbor_scalar(R, nbrs), neighbor_higher(R, nbrs))
#         assert torch.allclose(neighbor_scalar(R, nbrs), neighbor_tree(R, nbrs))


# @pytest.mark.parametrize("dtype, fmt", dtype_fmt_params)
# def test_pair_neighbor_list_per_atom_pytree(dtype, fmt):
#     torch.manual_seed(0)
#     dim = 2
#
#     def scalar_fn(dr: Tensor, sigma: Tensor, shift: Tensor) -> Tensor:
#         return torch.where(dr < sigma, dr ** 2 + shift, torch.tensor(0.0, dtype=torch.float32))
#
#     def higher_order_fn(dr: Tensor, p: Tensor) -> Tensor:
#         return torch.where(dr < p[..., 0], dr ** 2 + p[..., 1], torch.tensor(0.0, dtype=torch.float32))
#
#     @dataclasses.dataclass
#     class Parameter:
#         sigma: torch.Tensor
#         shift: torch.Tensor
#
#     def tree_fn(dr: Tensor, p):
#         return torch.where(dr < p.sigma, dr ** 2 + p.shift, torch.tensor(0.0, dtype=torch.float32))
#
#     N = NEIGHBOR_LIST_PARTICLE_COUNT
#     box_size = 4.0 * N ** (1.0 / dim)
#
#     displacement, _ = space(box=box_size, parallelepiped=False)
#     d = metric(displacement)
#
#     sigma = torch.rand((N,), dtype=dtype) * 0.5 + 0.5  # minval=0.5, maxval=1.0
#     shift = torch.rand((N,), dtype=dtype)
#     M = _ParameterTreeKind
#
#     neighbor_scalar = interact(scalar_fn, d, sigma=sigma, shift=shift, interaction="neighbor_list")
#     p = _ParameterTree(torch.cat([sigma[:, None], shift[:, None]], dim=-1), M.PARTICLE)
#     neighbor_higher = interact(higher_order_fn, d, p=p, interaction="neighbor_list")
#
#     p_tree = _ParameterTree(Parameter(sigma=sigma, shift=shift), M.PARTICLE)
#     neighbor_tree = interact(tree_fn, d, p=p_tree, interaction="neighbor_list")
#
#     for _ in range(STOCHASTIC_SAMPLES):
#         R = box_size * torch.rand((N, dim), dtype=dtype)
#         sigma = torch.rand(()) * 2.0 + 0.5  # minval=0.5, maxval=2.5
#         neighbor_fn = neighbor_list(displacement, torch.tensor([box_size] * dim, dtype=dtype), sigma, 0.0, neighbor_list_format=fmt)
#         nbrs = neighbor_fn.setup_fn(R)
#         assert torch.allclose(neighbor_scalar(R, nbrs), neighbor_higher(R, nbrs))
#         # assert torch.allclose(neighbor_scalar(R, nbrs), neighbor_tree(R, nbrs))
#
#
# @pytest.mark.parametrize("dtype, fmt", dtype_fmt_params)
# def test_pair_neighbor_list_per_atom_pytree(dtype, fmt):
#     torch.manual_seed(0)
#     dim = 2
#
#     def scalar_fn(dr: Tensor, sigma: Tensor, shift: Tensor) -> Tensor:
#         return torch.where(dr < sigma, dr ** 2 + shift, torch.tensor(0.0, dtype=torch.float32))
#
#     def higher_order_fn(dr: Tensor, p) -> Tensor:
#         return torch.where(dr < p[..., 0], dr ** 2 + p[..., 1], torch.tensor(0.0, dtype=torch.float32))
#
#     @dataclasses.dataclass
#     class Parameter:
#         sigma: torch.Tensor
#         shift: torch.Tensor
#
#     def tree_fn(dr: Tensor, p):
#         return torch.where(dr < p.sigma, dr ** 2 + p.shift, torch.tensor(0.0, dtype=torch.float32))
#
#     N = NEIGHBOR_LIST_PARTICLE_COUNT
#     box_size = 4.0 * N ** (1.0 / dim)
#
#     displacement, _ = space(box=box_size, parallelepiped=False)
#     d = metric(displacement)
#
#     sigma = torch.rand((N,), dtype=dtype) * 0.5 + 0.5  # minval=0.5, maxval=1.0
#     shift = torch.rand((N,), dtype=dtype)
#     M = _ParameterTreeKind
#
#     neighbor_scalar = interact(scalar_fn, d, sigma=sigma, shift=shift, interaction="neighbor_list")
#     p = _ParameterTree(torch.cat([sigma[:, None], shift[:, None]], dim=-1), M.PARTICLE)
#     neighbor_higher = interact(higher_order_fn, d, p=p, interaction="neighbor_list")
#
#     p_tree = _ParameterTree(Parameter(sigma=sigma, shift=shift), M.PARTICLE)
#     neighbor_tree = interact(tree_fn, d, p=p_tree, interaction="neighbor_list")
#
#     for _ in range(STOCHASTIC_SAMPLES):
#         R = box_size * torch.rand((N, dim), dtype=dtype)
#         sigma = torch.rand(()) * 2.0 + 0.5  # minval=0.5, maxval=2.5
#         neighbor_fn = neighbor_list(displacement, torch.tensor([box_size] * dim, dtype=dtype), sigma, 0.0, neighbor_list_format=fmt)
#         nbrs = neighbor_fn.setup_fn(R)
#         assert torch.allclose(neighbor_scalar(R, nbrs), neighbor_higher(R, nbrs))
#         # assert torch.allclose(neighbor_scalar(R, nbrs), neighbor_tree(R, nbrs))

# TODO (isaacsoh) broken due to safe indexing
# @pytest.mark.parametrize("dtype, fmt", dtype_fmt_params)
# def test_pair_neighbor_list_per_bond_pytree(dtype, fmt):
#     torch.manual_seed(0)
#     dim = 2
#
#     def scalar_fn(dr: Tensor, sigma: Tensor, shift: Tensor) -> Tensor:
#         return torch.where(dr < sigma, dr ** 2 + shift, torch.tensor(0.0, dtype=torch.float32))
#
#     def higher_order_fn(dr: Tensor, p) -> Tensor:
#         return torch.where(dr < p[..., 0], dr ** 2 + p[..., 1], torch.tensor(0.0, dtype=torch.float32))
#
#     @dataclasses.dataclass
#     class Parameter:
#         sigma: torch.Tensor
#         shift: torch.Tensor
#
#     def tree_fn(dr: Tensor, p) -> Tensor:
#         return torch.where(dr < p.sigma, dr ** 2 + p.shift, torch.tensor(0.0, dtype=torch.float32))
#
#     N = NEIGHBOR_LIST_PARTICLE_COUNT
#     box_size = 4.0 * N ** (1.0 / dim)
#
#     displacement, _ = space(box=box_size, parallelepiped=False)
#     d = metric(displacement)
#
#     sigma = torch.rand((N, N), dtype=dtype) * 0.5 + 0.5  # minval=0.5, maxval=1.0
#     shift = torch.rand((N, N), dtype=dtype)
#     M = _ParameterTreeKind
#
#     neighbor_scalar = interact(scalar_fn, d, sigma=sigma, shift=shift, interaction="neighbor_list")
#     p = _ParameterTree(torch.cat([sigma[:, :, None], shift[:, :, None]], dim=-1), M.BOND)
#     neighbor_higher = interact(higher_order_fn, d, p=p, interaction="neighbor_list")
#
#     p_tree = _ParameterTree(Parameter(sigma=sigma, shift=shift), M.BOND)
#     neighbor_tree = interact(tree_fn, d, p=p_tree, interaction="neighbor_list")
#
#     for _ in range(STOCHASTIC_SAMPLES):
#         R = box_size * torch.rand((N, dim), dtype=dtype)
#         sigma = torch.rand(()) * 2.0 + 0.5  # minval=0.5, maxval=2.5
#         neighbor_fn = neighbor_list(displacement, torch.tensor([box_size] * dim, dtype=dtype), sigma, 0.0, neighbor_list_format=fmt)
#         nbrs = neighbor_fn.setup_fn(R)
#         assert torch.allclose(neighbor_scalar(R, nbrs), neighbor_higher(R, nbrs))
#         assert torch.allclose(neighbor_scalar(R, nbrs), neighbor_tree(R, nbrs))


# @pytest.mark.parametrize("dtype, fmt", dtype_fmt_params)
# def test_pair_neighbor_list_kinds_global_pytree(dtype, fmt):
#     torch.manual_seed(0)
#     dim = 2
#
#     def scalar_fn(dr: Tensor, sigma: Tensor, shift: Tensor) -> Tensor:
#         return torch.where(dr < sigma, dr ** 2 + shift, torch.tensor(0.0, dtype=torch.float32))
#
#     def higher_order_fn(dr: Tensor, p) -> Tensor:
#         return torch.where(dr < p[0], dr ** 2 + p[1], torch.tensor(0.0, dtype=torch.float32))
#
#     @dataclasses.dataclass
#     class Parameter:
#         sigma: Tensor
#         shift: Tensor
#
#     def tree_fn(dr: Tensor, p) -> Tensor:
#         return torch.where(dr < p.sigma, dr ** 2 + p.shift, torch.tensor(0.0, dtype=torch.float32))
#
#     N = NEIGHBOR_LIST_PARTICLE_COUNT
#     box_size = 4.0 * N ** (1.0 / dim)
#
#     displacement, _ = space(box=box_size, parallelepiped=False)
#     d = metric(displacement)
#
#     sigma = torch.tensor(1.5, dtype=torch.float32)
#     shift = torch.tensor(2.0, dtype=torch.float32)
#     kinds = torch.where(torch.arange(N) < N // 2, 0, 1)
#     M = _ParameterTreeKind
#
#     neighbor_scalar = interact(scalar_fn, d, kinds=kinds, sigma=sigma, shift=shift, interaction="neighbor_list")
#     p = _ParameterTree(torch.tensor([sigma, shift]), M.SPACE)
#     neighbor_higher = interact(higher_order_fn, d, kinds=kinds, p=p, interaction="neighbor_list")
#
#     p_tree = _ParameterTree(Parameter(sigma=sigma, shift=shift), M.SPACE)
#     neighbor_tree = interact(tree_fn, d, kinds=kinds, p=p_tree, interaction="neighbor_list")
#
#     for _ in range(STOCHASTIC_SAMPLES):
#         R = box_size * torch.rand((N, dim), dtype=dtype)
#         sigma = torch.rand(()) * 2.0 + 0.5  # minval=0.5, maxval=2.5
#         neighbor_fn = neighbor_list(displacement, torch.tensor([box_size] * dim, dtype=dtype), sigma, 0.0, neighbor_list_format=fmt)
#         nbrs = neighbor_fn.setup_fn(R)
#         assert torch.allclose(neighbor_scalar(R, nbrs), neighbor_higher(R, nbrs))
#         assert torch.allclose(neighbor_scalar(R, nbrs), neighbor_tree(R, nbrs))

# TODO (isaacsoh) broken due to safe indexing
# @pytest.mark.parametrize("dtype, fmt", dtype_fmt_params)
# def test_pair_neighbor_list_kinds_per_kinds_pytree(dtype, fmt):
#     torch.manual_seed(0)
#     dim = 2
#
#     def scalar_fn(dr: Tensor, sigma: Tensor, shift: Tensor) -> Tensor:
#         return torch.where(dr < sigma, dr ** 2 + shift, torch.tensor(0.0, dtype=torch.float32))
#
#     def higher_order_fn(dr: Tensor, p):
#         return torch.where(dr < p[..., 0], dr ** 2 + p[..., 1], torch.tensor(0.0, dtype=torch.float32))
#
#     @dataclasses.dataclass
#     class Parameter:
#         sigma: Tensor
#         shift: Tensor
#
#     def tree_fn(dr: Tensor, p) -> Tensor:
#         return torch.where(dr < p.sigma, dr ** 2 + p.shift, torch.tensor(0.0, dtype=torch.float32))
#
#     N = NEIGHBOR_LIST_PARTICLE_COUNT
#     box_size = 4.0 * N ** (1.0 / dim)
#
#     displacement, _ = space(box=box_size, parallelepiped=False)
#     d = metric(displacement)
#
#     sigma = torch.tensor([[1.0, 1.2], [1.2, 1.5]], dtype=torch.float32)
#     shift = torch.tensor([[2.0, 1.5], [1.5, 3.0]], dtype=torch.float32)
#     kinds = torch.where(torch.arange(N) < N // 2, 0, 1)
#     M = _ParameterTreeKind
#
#     neighbor_scalar = interact(scalar_fn, d, kinds=kinds, sigma=sigma, shift=shift, interaction="neighbor_list")
#     p = _ParameterTree(torch.cat([sigma[..., None], shift[..., None]], dim=-1), M.KINDS)
#     neighbor_higher = interact(higher_order_fn, d, kinds=kinds, p=p, interaction="neighbor_list")
#
#     p_tree = _ParameterTree(Parameter(sigma=sigma, shift=shift), M.KINDS)
#     neighbor_tree = interact(tree_fn, d, kinds=kinds, p=p_tree, interaction="neighbor_list")
#
#     for _ in range(STOCHASTIC_SAMPLES):
#         R = box_size * torch.rand((N, dim), dtype=dtype)
#         neighbor_fn = neighbor_list(displacement, torch.tensor([box_size] * dim, dtype=dtype), torch.max(sigma), 0.0, neighbor_list_format=fmt)
#         nbrs = neighbor_fn.setup_fn(R)
#         assert torch.allclose(neighbor_scalar(R, nbrs), neighbor_higher(R, nbrs))
#         assert torch.allclose(neighbor_scalar(R, nbrs), neighbor_tree(R, nbrs))


# TODO (isaacsoh) broken due to interact bug
# @pytest.mark.parametrize("dtype, dim, fmt", neighbor_list_params)
# def test_pair_neighbor_list_scalar_diverging_potential(dtype, dim, fmt):
#     torch.manual_seed(0)
#
#     def potential(dr: Tensor, sigma):
#         return torch.where(dr < sigma, dr ** -6, torch.tensor(0.0, dtype=torch.float32))
#
#     N = NEIGHBOR_LIST_PARTICLE_COUNT
#     box_size = 4.0 * N ** (1.0 / dim)
#
#     displacement, _ = space(box=box_size, parallelepiped=False)
#     d = metric(displacement)
#
#     neighbor_square = interact(potential, d, sigma=1.0, interaction="neighbor_list")
#     mapped_square = interact(potential, d, sigma=1.0, interaction="pair")
#
#     for _ in range(STOCHASTIC_SAMPLES):
#         R = box_size * torch.rand((N, dim), dtype=dtype)
#         sigma = torch.rand(()) * 2.0 + 0.5  # minval=0.5, maxval=2.5
#         neighbor_fn = neighbor_list(displacement, torch.tensor([box_size] * dim, dtype=dtype), sigma, 0.0, neighbor_list_format=fmt)
#         nbrs = neighbor_fn.setup_fn(R)
#         assert torch.allclose(mapped_square(R, sigma=sigma), neighbor_square(R, nbrs, sigma=sigma))


# TODO (isaacsoh) skipped - should I use leonard jones here
# @pytest.mark.parametrize("dtype, dim, fmt", neighbor_list_params)
# def test_pair_neighbor_list_force_scalar_diverging_potential(dtype, dim, fmt):
#     torch.manual_seed(0)
#
#     def potential(dr, sigma):
#         return torch.where(dr < sigma, dr ** -6, torch.tensor(0.0, dtype=torch.float32))
#
#     N = NEIGHBOR_LIST_PARTICLE_COUNT
#     box_size = 4.0 * N ** (1.0 / dim)
#
#     displacement, _ = space(box=box_size, parallelepiped=False)
#     d = metric(displacement)
#
#     neighbor_square = interact(potential, d, sigma=1.0, interaction="neighbor_list")
#     neighbor_square = quantity_force(neighbor_square)
#     mapped_square = quantity_force(interact(potential, d, sigma=1.0, interaction="pair"))
#
#     for _ in range(STOCHASTIC_SAMPLES):
#         R = box_size * torch.rand((N, dim), dtype=dtype)
#         sigma = torch.rand(()) * 4.0 + 0.5  # minval=0.5, maxval=4.5
#         neighbor_fn = neighbor_list(displacement, torch.tensor([box_size] * dim, dtype=dtype), sigma, 0.0, neighbor_list_format=fmt)
#         nbrs = neighbor_fn.setup_fn(R)
#         assert torch.allclose(mapped_square(R, sigma=sigma), neighbor_square(R, nbrs, sigma=sigma))

# @pytest.mark.parametrize("dtype, dim, fmt", neighbor_list_params)
# def test_pair_neighbor_list_scalar_params_no_kinds(dtype, dim, fmt):
#     torch.manual_seed(0)
#
#     def truncated_square(dr: Tensor, sigma):
#         return torch.where(dr < sigma, dr ** 2, torch.tensor(0.0, dtype=torch.float32))
#
#     N = NEIGHBOR_LIST_PARTICLE_COUNT
#     box_size = 2.0 * N ** (1.0 / dim)
#
#     displacement, _ = space(box=box_size, parallelepiped=False)
#     d = metric(displacement)
#
#     neighbor_square = interact(truncated_square, d, sigma=1.0, interaction="neighbor_list")
#     mapped_square = interact(truncated_square, d, sigma=1.0, interaction="pair")
#
#     for _ in range(STOCHASTIC_SAMPLES):
#         R = box_size * torch.rand((N, dim), dtype=dtype)
#         sigma = torch.rand((N,), dtype=dtype) * 1.0 + 0.5  # minval=0.5, maxval=1.5
#         neighbor_fn = neighbor_list(displacement, torch.tensor([box_size] * dim, dtype=dtype), torch.max(sigma), 0.0, neighbor_list_format=fmt)
#         nbrs = neighbor_fn.setup_fn(R)
#         assert torch.allclose(mapped_square(R, sigma=sigma), neighbor_square(R, nbrs, sigma=sigma))

# TODO (isaacsoh) broken due to safe indexing
# @pytest.mark.parametrize("dtype, dim, fmt", neighbor_list_params)
# def test_pair_neighbor_list_scalar_params_matrix(dtype, dim, fmt):
#     torch.manual_seed(0)
#
#     def truncated_square(dr: Tensor, sigma):
#         return torch.where(dr < sigma, dr ** 2, torch.tensor(0.0, dtype=torch.float32))
#
#     N = NEIGHBOR_LIST_PARTICLE_COUNT
#     box_size = 2.0 * N ** (1.0 / dim)
#
#     displacement, _ = space(box=box_size, parallelepiped=False)
#     d = metric(displacement)
#
#     neighbor_square = interact(truncated_square, d, sigma=1.0, interaction="neighbor_list")
#     mapped_square = interact(truncated_square, d, sigma=1.0, interaction="pair")
#
#     for _ in range(STOCHASTIC_SAMPLES):
#         R = box_size * torch.rand((N, dim), dtype=dtype)
#         sigma = torch.rand((N, N), dtype=dtype) * 1.0 + 0.5  # minval=0.5, maxval=1.5
#         sigma = 0.5 * (sigma + sigma.T)
#         neighbor_fn = neighbor_list(displacement, torch.tensor([box_size] * dim, dtype=dtype), torch.max(sigma), 0.0, neighbor_list_format=fmt)
#         nbrs = neighbor_fn.setup_fn(R)
#         assert torch.allclose(mapped_square(R, sigma=sigma), neighbor_square(R, nbrs, sigma=sigma))


# TODO (isaacsoh) broken due to safe indexing
# @pytest.mark.parametrize("dtype, dim, fmt", neighbor_list_params)
# def test_pair_neighbor_list_scalar_params_kinds(dtype, dim, fmt):
#     torch.manual_seed(0)
#
#     def truncated_square(dr: Tensor, sigma):
#         return torch.where(dr < sigma, dr ** 2, torch.tensor(0.0, dtype=torch.float32))
#
#     N = NEIGHBOR_LIST_PARTICLE_COUNT
#     box_size = 2.0 * N ** (1.0 / dim)
#     kinds = torch.zeros((N,), dtype=torch.int32)
#     kinds = torch.where(torch.arange(N) > N / 3, 1, kinds)
#     kinds = torch.where(torch.arange(N) > 2 * N / 3, 2, kinds)
#
#     displacement, _ = space(box=box_size, parallelepiped=False)
#     d = metric(displacement)
#
#     neighbor_square = interact(truncated_square, d, kinds=kinds, sigma=1.0, interaction="neighbor_list")
#     mapped_square = interact(truncated_square, d, kinds=kinds, sigma=1.0, interaction="pair")
#
#     for _ in range(STOCHASTIC_SAMPLES):
#         R = box_size * torch.rand((N, dim), dtype=dtype)
#         sigma = torch.rand((3, 3), dtype=dtype) * 1.0 + 0.5  # minval=0.5, maxval=1.5
#         sigma = 0.5 * (sigma + sigma.T)
#         neighbor_fn = neighbor_list(displacement, torch.tensor([box_size] * dim, dtype=dtype), torch.max(sigma), 0.0, neighbor_list_format=fmt)
#         nbrs = neighbor_fn.setup_fn(R)
#         assert torch.allclose(mapped_square(R, sigma=sigma), neighbor_square(R, nbrs, sigma=sigma))


# TODO (isaacsoh) broken due to safe indexing
# @pytest.mark.parametrize("dtype, dim, fmt", neighbor_list_params)
# def test_pair_neighbor_list_scalar_params_kinds_dynamic(dtype, dim, fmt):
#     torch.manual_seed(0)
#
#     def truncated_square(dr: Tensor, sigma: Tensor, **kwargs):
#         return torch.where(dr < sigma, dr ** 2, torch.tensor(0.0, dtype=torch.float32))
#
#     N = NEIGHBOR_LIST_PARTICLE_COUNT
#     box_size = 2.0 * N ** (1.0 / dim)
#     kinds = torch.zeros((N,), dtype=torch.int32)
#     kinds = torch.where(torch.arange(N) > N / 3, 1, kinds)
#     kinds = torch.where(torch.arange(N) > 2 * N / 3, 2, kinds)
#
#     displacement, _ = space(box=box_size, parallelepiped=False)
#     d = metric(displacement)
#
#     neighbor_square = interact(truncated_square, d, sigma=1.0, interaction="neighbor_list")
#     mapped_square = interact(truncated_square, d, kinds=kinds, sigma=1.0, interaction="pair")
#
#     for _ in range(STOCHASTIC_SAMPLES):
#         R = box_size * torch.rand((N, dim), dtype=dtype)
#         sigma = torch.rand((3, 3), dtype=dtype) * 1.0 + 0.5  # minval=0.5, maxval=1.5
#         sigma = 0.5 * (sigma + sigma.T)
#         neighbor_fn = neighbor_list(displacement, torch.tensor([box_size] * dim, dtype=dtype), torch.max(sigma), 0.0, neighbor_list_format=fmt)
#         nbrs = neighbor_fn.setup_fn(R)
#         assert torch.allclose(mapped_square(R, sigma=sigma), neighbor_square(R, nbrs, sigma=sigma, kinds=kinds))


@pytest.mark.parametrize("dtype, dim, fmt", neighbor_list_params)
def test_pair_neighbor_list_vector(dtype, dim, fmt):
    if str(fmt) == "_NeighborListFormat.ORDERED_SPARSE":
        pytest.skip('Vector valued pair_neighbor_list not supported.')
    torch.manual_seed(0)

    def truncated_square(dR, sigma):
        dr = torch.reshape(distance(dR), dR.shape[:-1] + (1,))
        return torch.where(dr < sigma, dR ** 2, torch.tensor(0.0, dtype=torch.float32))

    N = PARTICLE_COUNT
    box_size = 2.0 * N ** (1.0 / dim)

    displacement, _ = space(box=box_size, parallelepiped=False)

    neighbor_square = interact(truncated_square, displacement, sigma=1.0, dim=(1,), interaction="neighbor_list")
    mapped_square = interact(truncated_square, displacement, sigma=1.0, dim=(1,), interaction="pair")

    for _ in range(STOCHASTIC_SAMPLES):
        R = box_size * torch.rand((N, dim), dtype=dtype)
        sigma = torch.rand(()) * 1.0 + 0.5  # minval=0.5, maxval=1.5
        neighbor_fn = neighbor_list(displacement, torch.tensor([box_size] * dim, dtype=dtype), sigma, 0.0, neighbor_list_format=fmt)
        nbrs = neighbor_fn.setup_fn(R)
        assert torch.allclose(mapped_square(R, sigma=sigma), neighbor_square(R, nbrs, sigma=sigma))



# @pytest.mark.parametrize("dtype, dim, fmt", neighbor_list_params)
# def test_pair_neighbor_list_vector_nonadditive(dtype, dim, fmt):
#     if fmt == "OrderedSparse":
#         pytest.skip('Vector valued pair_neighbor_list not supported.')
#     torch.manual_seed(0)
#
#     def truncated_square(dR, sigma):
#         dr = space_distance(dR)
#         return torch.where(dr < sigma, dr ** 2, torch.tensor(0.0, dtype=torch.float32))
#
#     N = PARTICLE_COUNT
#     box_size = 2.0 * N ** (1.0 / dim)
#
#     displacement, _ = space(box=box_size, parallelepiped=False)
#
#     neighbor_square = interact(truncated_square, displacement, sigma=lambda x, y: x * y, reduce_axis=(1,), interaction="neighbor_list")
#     mapped_square = interact(truncated_square, displacement, sigma=1.0, reduce_axis=(1,), interaction="pair")
#
#     for _ in range(STOCHASTIC_SAMPLES):
#         R = box_size * torch.rand((N, dim), dtype=dtype)
#         sigma = torch.rand((N,), dtype=dtype) * 1.0 + 0.5  # minval=0.5, maxval=1.5
#         sigma_pair = sigma[:, None] * sigma[None, :]
#         neighbor_fn = neighbor_list(displacement, torch.tensor([box_size] * dim, dtype=dtype), torch.max(sigma) ** 2, 0.0, neighbor_list_format=fmt)
#         nbrs = neighbor_fn.setup_fn(R)
#         assert torch.allclose(mapped_square(R, sigma=sigma_pair), neighbor_square(R, nbrs, sigma=sigma))
#







