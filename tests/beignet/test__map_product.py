import torch

from beignet import map_product


def euclidean_distance(x, y):
    return torch.sqrt(torch.sum((x - y) ** 2))


def test_map_product():
    positions1 = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    positions2 = torch.tensor([[1.0, 1.0], [2.0, 2.0]])

    expected_distances = torch.tensor(
        [
            [torch.sqrt(torch.tensor(2.0)), torch.sqrt(torch.tensor(0.0))],
            [torch.sqrt(torch.tensor(8.0)), torch.sqrt(torch.tensor(2.0))],
        ]
    )

    mapped_distance_fn = map_product(euclidean_distance)

    distances = mapped_distance_fn(positions1, positions2)

    assert torch.allclose(
        distances, expected_distances
    ), f"Expected {expected_distances}, but got {distances}"
