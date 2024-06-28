import pytest
import torch

from functools import partial
from torch import Tensor, vmap
from typing import Callable

from beignet.func._partition import metric, neighbor_list, safe_index, \
    map_product, _NeighborListFormat, neighbor_list_mask, _map_bond
from src.beignet.func import space

PARTICLE_COUNT = 1000
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
def test_neighbor_list_build(dtype, dim):
    torch.manual_seed(1)

    box_size = (
        torch.tensor([9.0, 4.0, 7.25], dtype=torch.float32)
        if dim == 3
        else torch.tensor([9.0, 4.25], dtype=torch.float32)
    )
    cutoff = torch.tensor(1.23, dtype=torch.float32)

    displacement, _ = space(box=box_size, parallelepiped=False)

    metric_fn = metric(displacement)

    R = box_size * torch.rand((PARTICLE_COUNT, dim), dtype=dtype)
    N = R.shape[0]

    neighbor_fn = neighbor_list(displacement, box_size, cutoff, 0.0, 1.1)

    idx = neighbor_fn.setup_fn(R).indexes

    R_neigh = safe_index(R, idx)

    mask = idx < N

    d = vmap(vmap(metric_fn, in_dims=(None, 0)))

    dR = d(R, R_neigh)

    d_exact = map_product(metric_fn)
    dR_exact = d_exact(R, R)

    dR = torch.where(dR < cutoff, dR, torch.tensor(0, dtype=torch.float32)) * mask

    mask_exact = 1.0 - torch.eye(dR_exact.shape[0])
    dR_exact = (
        torch.where(dR_exact < cutoff, dR_exact, torch.tensor(0, dtype=torch.float32))
        * mask_exact
    )

    dR, _ = torch.sort(dR, dim=1)
    dR_exact, _ = torch.sort(dR_exact, dim=1)

    for i in range(dR.shape[0]):
        dR_row = dR[i]
        dR_row = dR_row[dR_row > 0.0]

        dR_exact_row = dR_exact[i]
        dR_exact_row = torch.tensor(dR_exact_row[dR_exact_row > 0.0], dtype=dtype)

        assert torch.allclose(dR_row, dR_exact_row)


@pytest.mark.parametrize("dtype, dim", params)
def test_neighbor_list_build_sparse(dtype, dim):
    torch.manual_seed(1)

    box_size = (
        torch.tensor([9.0, 4.0, 7.25], dtype=torch.float32)
        if dim == 3
        else torch.tensor([9.0, 4.25], dtype=torch.float32)
    )
    cutoff = torch.tensor(1.23, dtype=torch.float32)

    displacement, _ = space(box=box_size, parallelepiped=False)
    metric_fn = metric(displacement)

    R = box_size * torch.rand((16, dim), dtype=dtype)
    N = R.shape[0]

    neighbor_fn = neighbor_list(
        displacement,
        box_size,
        cutoff,
        0.0,
        1.1,
        neighbor_list_format=_NeighborListFormat.SPARSE,
    )

    nbrs = neighbor_fn.setup_fn(R)
    mask = neighbor_list_mask(nbrs)

    d = _map_bond(metric_fn)
    dR = d(safe_index(R, nbrs.indexes[0]), safe_index(R, nbrs.indexes[1]))

    d_exact = map_product(metric_fn)
    dR_exact = d_exact(R, R)

    dR = torch.where(dR < cutoff, dR, torch.tensor(0)) * mask
    mask_exact = 1.0 - torch.eye(dR_exact.shape[0])
    dR_exact = torch.where(dR_exact < cutoff, dR_exact, torch.tensor(0)) * mask_exact

    dR_exact, _ = torch.sort(dR_exact, dim=1)

    for i in range(N):
        dR_row = dR[nbrs.indexes[0] == i]
        dR_row = dR_row[dR_row > 0.0]
        dR_row, _ = torch.sort(dR_row)

        dR_exact_row = dR_exact[i]
        dR_exact_row = torch.tensor(dR_exact_row[dR_exact_row > 0.0], dtype=dtype)

        assert torch.allclose(dR_row, dR_exact_row)


def test_cell_list_overflow():
    displacement_fn, shift_fn = space()

    box = torch.tensor(100.0)
    r_cutoff = 3.0
    dr_threshold = 0.0

    neighbor_fn = neighbor_list(
        displacement_fn=displacement_fn,
        space=box,
        neighborhood_radius=r_cutoff,
        maximum_distance=dr_threshold,
    )

    # all far from eachother
    positions = torch.tensor(
        [
            [20.0, 20.0],
            [30.0, 30.0],
            [40.0, 40.0],
            [50.0, 50.0],
        ]
    )

    neighbors = neighbor_fn.setup_fn(positions)

    assert neighbors.indexes.dtype is torch.int32

    # two first point are close to eachother
    positions = torch.tensor(
        [
            [20.0, 20.0],
            [20.0, 20.0],
            [40.0, 40.0],
            [50.0, 50.0],
        ]
    )

    neighbors = neighbor_fn.update_fn(positions, neighbors)

    assert neighbors.did_buffer_overflow
    assert neighbors.indexes.dtype is torch.int32


def test_custom_mask_function():
    displacement_fn, shift_fn = space()

    box = torch.tensor(1.0)
    r_cutoff = 3.0
    dr_threshold = 0.0
    n_particles = 10
    R = torch.zeros(3).expand(n_particles, 3)

    def acceptable_id_pair(id1, id2):
        """
        Don't allow particles to have an interaction when their id's
        are closer than 3 (eg disabling 1-2 and 1-3 interactions)
        """
        return torch.abs(id1 - id2) > 3

    def mask_id_based(
        idx: Tensor, ids: Tensor, mask_val: int, _acceptable_id_pair: Callable
    ) -> Tensor:
        """
        _acceptable_id_pair mapped to act upon the neighbor list where:
        - index of particle 1 is in index in the first dimension of array
        - index of particle 2 is given by the value in the array
        """

        @partial(vmap, in_dims=(0, 0, None))
        def acceptable_id_pair(idx, id1, ids):
            id2 = safe_index(ids, idx)

            return vmap(_acceptable_id_pair, in_dims=(None, 0))(id1, id2)

        mask = acceptable_id_pair(idx, ids, ids)

        return torch.where(mask, idx, mask_val)

    ids = torch.arange(n_particles)  # id is just particle index here.
    mask_val = n_particles
    custom_mask_function = partial(
        mask_id_based,
        ids=ids,
        mask_val=mask_val,
        _acceptable_id_pair=acceptable_id_pair,
    )

    neighbor_fn = neighbor_list(
        displacement_fn=displacement_fn,
        space=box,
        neighborhood_radius=r_cutoff,
        maximum_distance=dr_threshold,
        mask_fn=custom_mask_function,
    )

    neighbors = neighbor_fn.setup_fn(R)
    neighbors = neighbors.update_fn(R)
    """
    Without masking it's 9 neighbors (with mask self) -> 90 neighbors.
    With masking -> 42.
    """
    assert 42 == (neighbors.indexes != mask_val).sum()


def test_issue191_1():
    box_vector = torch.ones(3) * 3

    r_cut = 0.1
    _positions = torch.linspace(0.5, 0.7, 20)
    positions = torch.stack([_positions, _positions, _positions], dim=1)

    displacement, _ = space(box_vector, parallelepiped=True)

    neighbor_fn = neighbor_list(
        displacement,
        box_vector,
        r_cut,
        0.1 * r_cut,
        normalized=True,
    )

    neighbor2_fn = neighbor_list(
        displacement,
        box_vector[0],
        r_cut,
        0.1 * r_cut,
        normalized=True,
        disable_unit_list=True,
    )

    nbrs = neighbor_fn.setup_fn(positions)
    nbrs2 = neighbor2_fn.setup_fn(positions)

    tensor_1, _ = torch.sort(nbrs.indexes, dim=-1)
    tensor_2, _ = torch.sort(nbrs2.indexes, dim=-1)

    assert torch.allclose(tensor_1, tensor_2)


@pytest.mark.parametrize(
    "r_cut, disable_cell_list, capacity_multiplier, mask_self, fmt",
    [
        (0.12, True, 1.5, False, _NeighborListFormat.DENSE),
        (0.12, True, 1.5, False, _NeighborListFormat.SPARSE),
        (0.12, True, 1.5, False, _NeighborListFormat.ORDERED_SPARSE),
        (0.12, True, 1.5, True, _NeighborListFormat.DENSE),
        (0.12, True, 1.5, True, _NeighborListFormat.SPARSE),
        (0.12, True, 1.5, True, _NeighborListFormat.ORDERED_SPARSE),
        (0.25, False, 1.5, False, _NeighborListFormat.DENSE),
        (0.25, False, 1.5, False, _NeighborListFormat.SPARSE),
        (0.25, False, 1.5, False, _NeighborListFormat.ORDERED_SPARSE),
        (0.25, False, 1.5, True, _NeighborListFormat.DENSE),
        (0.25, False, 1.5, True, _NeighborListFormat.SPARSE),
        (0.25, False, 1.5, True, _NeighborListFormat.ORDERED_SPARSE),
        (0.31, False, 1.5, False, _NeighborListFormat.DENSE),
        (0.31, False, 1.5, False, _NeighborListFormat.SPARSE),
        (0.31, False, 1.5, False, _NeighborListFormat.ORDERED_SPARSE),
        (0.31, False, 1.5, True, _NeighborListFormat.DENSE),
        (0.31, False, 1.5, True, _NeighborListFormat.SPARSE),
        (0.31, False, 1.5, True, _NeighborListFormat.ORDERED_SPARSE),
        (0.31, False, 1.0, False, _NeighborListFormat.DENSE),
        (0.31, False, 1.0, False, _NeighborListFormat.SPARSE),
        (0.31, False, 1.0, False, _NeighborListFormat.ORDERED_SPARSE),
        (0.31, False, 1.0, True, _NeighborListFormat.DENSE),
        (0.31, False, 1.0, True, _NeighborListFormat.SPARSE),
        (0.31, False, 1.0, True, _NeighborListFormat.ORDERED_SPARSE),
    ],
)
def test_issue191_2(r_cut, disable_cell_list, capacity_multiplier, mask_self, fmt):
    box = torch.ones(3)
    # box = 1.0
    if fmt is _NeighborListFormat.DENSE:
        desired_shape = (20, 19) if mask_self else (20, 20)

        _positions = torch.ones((20,)) * 0.5

    elif fmt is _NeighborListFormat.SPARSE:
        desired_shape = (2, 20 * 19) if mask_self else (2, 20**2)

        _positions = torch.ones((20,)) * 0.5

    elif fmt is _NeighborListFormat.ORDERED_SPARSE:
        desired_shape = (2, 20 * 19 // 2)

        _positions = torch.ones((20,)) * 0.5

    positions = torch.stack([_positions, _positions, _positions], dim=1)

    displacement, _ = space(box=box, parallelepiped=False)

    neighbor_fn = neighbor_list(
        displacement,
        box,
        r_cut,
        0.1 * r_cut,
        buffer_size_multiplier=capacity_multiplier,
        disable_unit_list=disable_cell_list,
        mask_self=mask_self,
        neighbor_list_format=fmt,
    )

    nbrs = neighbor_fn.setup_fn(positions)

    assert nbrs.did_buffer_overflow is False
    assert nbrs.indexes.shape == desired_shape

    new_nbrs = neighbor_fn.update_fn(positions + 0.1, nbrs)

    assert new_nbrs.did_buffer_overflow is False
    assert new_nbrs.indexes.shape == desired_shape
