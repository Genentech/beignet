import torch
import pytest

from beignet.dipole_moments import dipole_moments


def test_basic_dipole_moments():
    # This test case assumes a simple setup where the math can be easily verified.
    positions = torch.tensor(
        [[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]],
        dtype=torch.float32,
    )  # Two frames, two atoms
    charges = torch.tensor(
        [1.0, -1.0], dtype=torch.float32
    )  # Positive and negative charges

    expected_dipoles = torch.tensor(
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float32
    )
    computed_dipoles = dipole_moments(positions, charges)

    assert torch.allclose(
        computed_dipoles, expected_dipoles
    ), "Basic dipole moments calculation failed."


def test_zero_charges():
    # Tests the scenario where all charges are zero - resulting in zero dipole moment.
    positions = torch.randn(1, 5, 3)  # One frame, five atoms, arbitrary positions
    charges = torch.zeros(5, dtype=torch.float32)  # Zero charges

    expected_dipoles = torch.zeros(1, 3, dtype=torch.float32)
    computed_dipoles = dipole_moments(positions, charges)

    assert torch.allclose(
        computed_dipoles, expected_dipoles
    ), "Dipole moment should be zero when all charges are zero."


def test_negative_and_positive_charges():
    # Mixed charges but symmetrically arranged atoms ensuring zero dipole moment
    positions = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float32)
    charges = torch.tensor([1.0, -1.0], dtype=torch.float32)

    expected_dipoles = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    computed_dipoles = dipole_moments(positions.unsqueeze(0), charges)

    assert torch.allclose(
        computed_dipoles, expected_dipoles
    ), "Dipole moment should be zero for symmetric charges and positions."
