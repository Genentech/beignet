import unittest

import pytest
import torch
from functorch import vmap

from src.beignet.func import space
from src.beignet.func._molecular_dynamics._partition.__clamp_indices import \
    clamp_indices
from src.beignet.func._molecular_dynamics._partition._neighbor_list import \
    neighbor_list
from src.beignet.func._molecular_dynamics._partition.__metric import metric
from src.beignet.func._molecular_dynamics._partition.__map_product import map_product

PARTICLE_COUNT = 1000
SPATIAL_DIMENSION = [2]


@pytest.mark.parametrize(
    "dim", [(dim) for dim in SPATIAL_DIMENSION]
)
def test_neighbor_list_build(dim):
    torch.manual_seed(1)

    box_size = (
        torch.tensor([9.0, 4.0, 7.25]) if dim == 3 else
        torch.tensor([9.0, 4.25]))
    cutoff = torch.tensor(1.23)

    displacement, _ = space(box=box_size, parallelepiped=False)

    metric_fn = metric(displacement)

    R = box_size * torch.rand((PARTICLE_COUNT, dim)) # R.shape =  torch.Size([1000, 2])
    N = R.shape[0]

    neighbor_fn = neighbor_list(
        displacement, box_size, cutoff, 0.0, 1.1)

    idx = neighbor_fn.setup_fn(R).indexes # idx.shape = torch.Size([1000, 85])

    try:
        R_neigh = R[idx] # R_neigh.shape = torch.Size([1000, 85, 2])

    except:
        R_neigh = torch.zeros(idx.shape[0], idx.shape[1], R.shape[1])

    mask = idx < N # mask.shape == torch.Size([1000, 85])

    d = vmap(vmap(metric_fn, in_dims=(None, 0)))

    assert R.shape == (1000, 2)
    assert R_neigh.shape == (1000, 479, 2)

    dR = d(R, R_neigh) # dR.shape = torch.Size([1000, 479, 2, 2])

    # assert torch.equal(dR, torch.tensor([1, 2, 3]))

    assert dR.shape == (1000, 479)

    d_exact = map_product(metric_fn)
    dR_exact = d_exact(R, R)

    multiplier = torch.where(dR < cutoff, dR, torch.tensor(0))

    # assert multiplier.shape == (1) # multiplier.shape = torch.Size([1000, 85, 2, 2])

    dR = multiplier * mask # turns it to torch.zeros i think

    assert torch.equal(dR, torch.tensor([1, 2, 3]))

    mask_exact = 1. - torch.eye(dR_exact.shape[0])
    dR_exact = torch.where(dR_exact < cutoff, dR_exact, torch.tensor(0)) * mask_exact

    dR, _ = torch.sort(dR, dim=1)
    dR_exact, _ = torch.sort(dR_exact, dim=1)

    for i in range(dR.shape[0]):
        dR_row = dR[i]

        # assert dR_row.shape == (479)

        assert torch.equal(dR, torch.tensor([1, 2, 3]))



        dR_row = dR_row[dR_row > 0.]

        assert dR_row.shape == (10, 12)

        dR_exact_row = dR_exact[i]
        dR_exact_row = torch.tensor(dR_exact_row[dR_exact_row > 0.])

        assert torch.allclose(dR_row, dR_exact_row)
