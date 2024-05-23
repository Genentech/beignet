import torch
import pytest

from beignet.dihedrals import dihedrals


def radians(degrees):
    """Utility function to convert degrees to radians."""
    return degrees * torch.pi / 180


def test_dihedral_180_degrees():
    # Tests four collinear points which should result in a dihedral angle of 180 degrees (pi radians)
    traj = torch.tensor(
        [[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]], dtype=torch.float32
    )
    indices = torch.tensor([[0, 1, 2, 3]])

    expected_angles = torch.tensor([[radians(180)]])
    computed_angles = dihedrals(traj, indices)

    assert torch.allclose(
        computed_angles, expected_angles
    ), "Dihedral angle should be 180 degrees for collinear points."


def test_dihedral_90_degrees():
    # Configuration that should result in a dihedral angle of 90 degrees
    traj = torch.tensor(
        [[[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1]]], dtype=torch.float32
    )  # A right-angle turn at the second atom
    indices = torch.tensor([[0, 1, 2, 3]])

    expected_angles = torch.tensor([[radians(90)]])
    computed_angles = dihedrals(traj, indices)

    assert torch.allclose(
        torch.abs(computed_angles), expected_angles, atol=1e-5
    ), "Dihedral angle should be 90 degrees."


def test_no_indices_provided():
    # Providing no indices should return an empty tensor
    traj = torch.randn(1, 10, 3)
    indices = torch.empty((0, 4), dtype=torch.int32)

    computed_angles = dihedrals(traj, indices)
    assert (
        computed_angles.nelement() == 0
    ), "Providing no dihedral indices should result in an empty tensor."


def test_index_out_of_bounds():
    # Providing indices out of bounds should raise an error
    traj = torch.randn(1, 4, 3)
    indices = torch.tensor([[0, 1, 5, 3]])  # Index 5 is out of bounds

    with pytest.raises(ValueError):
        dihedrals(traj, indices)


if __name__ == "__main__":
    pytest.main()
