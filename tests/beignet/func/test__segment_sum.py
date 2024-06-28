import torch

from beignet.func._partition import _segment_sum


def test_segment_sum():
    one_particle_hash = torch.tensor([[1]])
    particle_hash = torch.tensor([[6]])
    cell_count = 4

    assert torch.equal(
        _segment_sum(one_particle_hash, particle_hash, cell_count),
        torch.tensor([[0], [0], [0], [0]]),
    )
