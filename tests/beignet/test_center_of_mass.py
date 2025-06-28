import torch

from beignet.center_of_mass import center_of_mass


def test_center_of_mass_basic():
    positions = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]],
        ]
    )
    masses = torch.tensor([1.0, 1.0])

    expected_com = torch.tensor(
        [
            [0.5, 0.5, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )

    com = center_of_mass(positions, masses)

    assert torch.allclose(com, expected_com)


def test_center_of_mass_shape():
    positions = torch.randn(10, 5, 3)
    masses = torch.rand(5)

    com = center_of_mass(positions, masses)

    assert com.shape == (10, 3)


def test_center_of_mass_at_origin():
    positions = torch.zeros(3, 4, 3)
    masses = torch.rand(4)

    com = center_of_mass(positions, masses)

    assert torch.all(com == 0)
