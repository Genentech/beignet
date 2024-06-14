import torch

from src.beignet.func._molecular_dynamics._partition.__particles_per_cell import \
    _particles_per_cell


def test__particles_per_cell():
    positions = torch.tensor([
        [0.5],
        [1.5],
        [2.5],
        [3.5]
    ])

    box_size = torch.tensor([4.0])
    minimum_cell_size = 1.0

    assert torch.equal(
        _particles_per_cell(positions, box_size, minimum_cell_size),
        torch.tensor([[0], [0], [0], [0]])
    )

