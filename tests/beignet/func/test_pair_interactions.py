import dataclasses

import pytest
import torch
from torch import Tensor

from beignet.func import space
from beignet.func._interact import interact, _ParameterTreeKind, _ParameterTree
from beignet.func._partition import map_product, metric

PARTICLE_COUNT = 1000
STOCHASTIC_SAMPLES = 3
POSITION_DTYPE = [torch.float32, torch.float64]  # Example values
SPATIAL_DIMENSION = [2, 3]
test_cases = [
    {
        "dtype": dtype,
        "dim": dim,
    }
    for dtype in POSITION_DTYPE
    for dim in SPATIAL_DIMENSION
]

# Extract the parameters and ids
params = [(case["dtype"], case["dim"]) for case in test_cases]


@pytest.mark.parametrize("dtype, dim", params)
def test_pair_no_species_scalar(dtype, dim):
    square = lambda dr: dr ** 2

    displacement, _ = space(box=None)

    metric = lambda Ra, Rb, **kwargs: \
        torch.sum(displacement(Ra, Rb, **kwargs) ** 2, dim=-1)

    mapped_square = interact(
        fn=square,
        displacement_fn=metric,
        interaction="pair",
    )

    metric = map_product(metric)

    torch.manual_seed(0)

    for _ in range(STOCHASTIC_SAMPLES):
        R = torch.rand((PARTICLE_COUNT, dim), dtype=dtype)

        assert torch.allclose(
            mapped_square(R),
            torch.tensor(0.5 * torch.sum(square(metric(R, R))), dtype=dtype)
        )


@pytest.mark.parametrize("dtype", POSITION_DTYPE)
def test_pair_no_species_pytree(dtype):
    square_scalar = lambda dr, p0, p1: p0 * dr ** 2 + p1
    square_higher = lambda dr, p: p[0] * dr ** 2 + p[1]

    @dataclasses.dataclass
    class Parameter:
        scale: Tensor
        shift: Tensor

    tree_fn = lambda dr, p: p.scale * dr ** 2 + p.shift
    displacement, _ = space(box=None)
    test_metric = metric(displacement)

    p = torch.tensor([1.0, 2.0])
    M = _ParameterTreeKind
    mapped_scalar = interact(
        square_scalar,
        test_metric,
        interaction="pair",
        p0=p[0],
        p1=p[1],
    )
    mapped_higher = interact(
        square_higher,
        test_metric,
        interaction="pair",
        p=_ParameterTree(p, M.SPACE)
    )

    p_tree = _ParameterTree(Parameter(scale=p[0], shift=p[1]), M.SPACE)
    mapped_tree = interact(tree_fn, test_metric, p=p_tree, interaction="pair")

    torch.manual_seed(0)

    for _ in range(STOCHASTIC_SAMPLES):
        R = torch.rand((PARTICLE_COUNT, 2), dtype=dtype)

        assert torch.allclose(mapped_scalar(R), mapped_higher(R))
        assert torch.allclose(mapped_scalar(R), mapped_tree(R))


@pytest.mark.parametrize("dtype, dim", params)
def test_pair_no_species_scalar_dynamic(dtype, dim):
    square = lambda dr, epsilon: epsilon * dr ** 2
    displacement, _ = space(box=None)
    metric = lambda Ra, Rb, **kwargs: \
        torch.sum(displacement(Ra, Rb, **kwargs) ** 2, dim=-1)

    mapped_square = interact(square, metric, epsilon=1.0, interaction="pair")
    metric = map_product(metric)

    torch.manual_seed(0)

    for _ in range(STOCHASTIC_SAMPLES):
        R = torch.rand((PARTICLE_COUNT, dim), dtype=dtype)
        epsilon = torch.rand((PARTICLE_COUNT,), dtype=dtype)
        mat_epsilon = 0.5 * (epsilon[:, None] + epsilon[None, :])

        assert torch.allclose(
            mapped_square(R, epsilon=epsilon),
            torch.tensor(
                0.5 * torch.sum(square(metric(R, R), mat_epsilon)), dtype=dtype
            )
        )


@pytest.mark.parametrize("dtype, dim", params)
def test_pair_no_species_vector(dtype, dim):
    square = lambda dr: torch.sum(dr ** 2, dim=2)
    disp, _ = space(box=None)

    mapped_square = interact(square, disp, interaction="pair")

    disp = map_product(disp)
    torch.manual_seed(0)

    for _ in range(STOCHASTIC_SAMPLES):
        R = torch.rand((PARTICLE_COUNT, dim), dtype=dtype)
        mapped_ref = torch.tensor(0.5 * torch.sum(square(disp(R, R))),
                                  dtype=dtype)

        assert torch.allclose(mapped_square(R), mapped_ref)


# @pytest.mark.parametrize("dtype", POSITION_DTYPE)
# def test_pair_no_species_pytree_per_particle(dtype):
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
#     for _ in range(STOCHASTIC_SAMPLES):
#         R = torch.rand((PARTICLE_COUNT, 2), dtype=dtype)
#         assert torch.allclose(mapped_scalar(R), mapped_higher(R))
#         assert torch.allclose(mapped_scalar(R), mapped_tree(R))



