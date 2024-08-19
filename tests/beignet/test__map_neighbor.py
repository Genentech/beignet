import torch

from beignet import map_neighbor


def euclidean_distance(x, y):
    return torch.sqrt(torch.sum((x - y) ** 2))


def test_map_neighbor():
    reference_positions = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    neighborhood_positions = torch.tensor([[[1.0, 1.0], [2.0, 2.0]], [[2.0, 2.0], [3.0, 3.0]]])

    expected_distances = torch.tensor([
        [torch.sqrt(torch.tensor(2.0)), torch.sqrt(torch.tensor(8.0))],
        [torch.sqrt(torch.tensor(2.0)), torch.sqrt(torch.tensor(8.0))]
    ])

    mapped_distance_fn = map_neighbor(euclidean_distance)

    distances = mapped_distance_fn(reference_positions, neighborhood_positions)

    assert torch.allclose(distances, expected_distances), f"Expected {expected_distances}, but got {distances}"
