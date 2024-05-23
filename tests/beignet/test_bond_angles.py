import torch
import pytest

from beignet.bond_angles import bond_angles


def radians(degrees):
    """Utility function to convert degrees to radians."""
    return degrees * torch.pi / 180


def test_straight_line_angle():
    # Tests three collinear points which must produce an angle of pi radians (180 degrees)
    traj = torch.tensor([[[0, 0, 0], [1, 0, 0], [2, 0, 0]]], dtype=torch.float32)
    angle_indices = torch.tensor([[0, 1, 2]])

    expected_angles = torch.tensor([[radians(180)]])
    computed_angles = bond_angles(traj, angle_indices)

    assert torch.allclose(
        computed_angles, expected_angles
    ), "Should calculate 180 degrees for collinear points."


def test_right_angle():
    # Tests an L shape (right angle, 90 degrees)
    traj = torch.tensor([[[0, 0, 0], [1, 0, 0], [1, 1, 0]]], dtype=torch.float32)
    angle_indices = torch.tensor([[0, 1, 2]])

    expected_angles = torch.tensor([[radians(90)]])
    computed_angles = bond_angles(traj, angle_indices)

    assert torch.allclose(
        computed_angles, expected_angles, atol=1e-5
    ), "Should calculate 90 degrees for orthogonal vectors."


def test_acute_angle():
    # Acute angle test 45 degrees
    traj = torch.tensor(
        [[[0, 0, 0], [1, 0, 0], [1, torch.sqrt(torch.tensor(2.0)), 0]]],
        dtype=torch.float32,
    )
    angle_indices = torch.tensor([[0, 1, 2]])

    expected_angles = torch.tensor([[radians(45)]])
    computed_angles = bond_angles(traj, angle_indices)

    assert torch.allclose(
        computed_angles, expected_angles, atol=1e-5
    ), "Should calculate 45 degrees for acute angle."


def test_no_indices_provided():
    # Providing no indices should return an empty tensor
    traj = torch.randn(1, 10, 3)
    angle_indices = torch.empty((0, 3), dtype=torch.int32)

    computed_angles = bond_angles(traj, angle_indices)
    assert (
        computed_angles.nelement() == 0
    ), "Providing no indices should result in an empty tensor."
