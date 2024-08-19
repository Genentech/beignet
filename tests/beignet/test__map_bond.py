import torch

from beignet import map_bond


def euclidean_distance(x, y):
    return torch.sqrt(torch.sum((x - y) ** 2))


def test_map_bond():
    start_positions = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    end_positions = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    expected_distances = torch.tensor(
        [
            torch.sqrt(torch.tensor(2.0)),
            torch.sqrt(torch.tensor(2.0)),
            torch.sqrt(torch.tensor(2.0)),
        ]
    )

    mapped_distance_fn = map_bond(euclidean_distance)

    distances = mapped_distance_fn(start_positions, end_positions)

    assert torch.allclose(
        distances, expected_distances
    ), f"Expected {expected_distances}, but got {distances}"
