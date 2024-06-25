import pytest
import torch
from functorch import vmap

from src.beignet.func import space
from src.beignet.func._molecular_dynamics._partition.__map_bond import \
    _map_bond
from src.beignet.func._molecular_dynamics._partition.__neighbor_list_format import \
    _NeighborListFormat
from src.beignet.func._molecular_dynamics._partition.__neighbor_list_mask import \
    neighbor_list_mask
from src.beignet.func._molecular_dynamics._partition.__safe_index import \
    safe_index
from src.beignet.func._molecular_dynamics._partition._neighbor_list import \
    neighbor_list
from src.beignet.func._molecular_dynamics._partition.__metric import metric
from src.beignet.func._molecular_dynamics._partition.__map_product import map_product

PARTICLE_COUNT = 1000
SPATIAL_DIMENSION = [2, 3]


# @pytest.mark.parametrize(
#     "dim", [(dim) for dim in SPATIAL_DIMENSION]
# )
# def test_neighbor_list_build(dim):
#     torch.manual_seed(1)
#
#     box_size = (
#         torch.tensor([9.0, 4.0, 7.25]) if dim == 3 else
#         torch.tensor([9.0, 4.25]))
#     cutoff = torch.tensor(1.23)
#
#     displacement, _ = space(box=box_size, parallelepiped=False)
#
#     metric_fn = metric(displacement)
#
#     R = box_size * torch.rand((PARTICLE_COUNT, dim)) # R.shape =  torch.Size([1000, 2])
#     N = R.shape[0]
#
#     neighbor_fn = neighbor_list(
#         displacement, box_size, cutoff, 0.0, 1.1)
#
#     idx = neighbor_fn.setup_fn(R).indexes
#
#     R_neigh = safe_index(R, idx)
#
#     mask = idx < N
#
#     d = vmap(vmap(metric_fn, in_dims=(None, 0)))
#
#     dR = d(R, R_neigh)
#
#     d_exact = map_product(metric_fn)
#     dR_exact = d_exact(R, R)
#
#     multiplier = torch.where(dR < cutoff, dR, torch.tensor(0))
#
#     dR = multiplier * mask
#
#     mask_exact = 1. - torch.eye(dR_exact.shape[0])
#     dR_exact = torch.where(dR_exact < cutoff, dR_exact, torch.tensor(0)) * mask_exact
#
#     dR, _ = torch.sort(dR, dim=1)
#     dR_exact, _ = torch.sort(dR_exact, dim=1)
#
#     for i in range(dR.shape[0]):
#         dR_row = dR[i]
#         dR_row = dR_row[dR_row > 0.]
#
#         dR_exact_row = dR_exact[i]
#         dR_exact_row = torch.tensor(dR_exact_row[dR_exact_row > 0.])
#
#         assert torch.allclose(dR_row, dR_exact_row)
#
#
# @pytest.mark.parametrize(
#     "dim", [(dim) for dim in SPATIAL_DIMENSION]
# )
# def test_neighbor_list_build_sparse(dim):
#     torch.manual_seed(1)
#
#     box_size = (
#         torch.tensor([9.0, 4.0, 7.25]) if dim == 3 else
#         torch.tensor([9.0, 4.25]))
#     cutoff = torch.tensor(1.23)
#
#     displacement, _ = space(box=box_size, parallelepiped=False)
#     metric_fn = metric(displacement)
#
#     R = box_size * torch.rand((1000, dim))
#     N = R.shape[0]
#
#     neighbor_fn = neighbor_list(
#         displacement, box_size, cutoff, 0.0, 1.1, neighbor_list_format=_NeighborListFormat.SPARSE)
#
#     nbrs = neighbor_fn.setup_fn(R)
#     mask = neighbor_list_mask(nbrs)
#
#     d = _map_bond(metric_fn)
#     dR = d(safe_index(R, nbrs.indexes[0]), safe_index(R, nbrs.indexes[1]))
#
#     d_exact = map_product(metric_fn)
#     dR_exact = d_exact(R, R)
#
#     dR = torch.where(dR < cutoff, dR, torch.tensor(0)) * mask
#     mask_exact = 1. - torch.eye(dR_exact.shape[0])
#     dR_exact = torch.where(dR_exact < cutoff, dR_exact, torch.tensor(0)) * mask_exact
#
#     dR_exact, _ = torch.sort(dR_exact, dim=1)
#
#     for i in range(N):
#       dR_row = dR[nbrs.indexes[0] == i]
#       dR_row = dR_row[dR_row > 0.]
#       dR_row, _ = torch.sort(dR_row)
#
#       dR_exact_row = dR_exact[i]
#       dR_exact_row = torch.tensor(dR_exact_row[dR_exact_row > 0.])
#
#       assert torch.allclose(dR_row, dR_exact_row)


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
