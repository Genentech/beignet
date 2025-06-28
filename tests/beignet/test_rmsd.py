import torch
from beignet.rmsd import rmsd


def test_rmsd_2d_case():
    traj1 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    traj2 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    expected_rmsd = torch.zeros(2)
    out = rmsd(traj1, traj2)
    assert torch.allclose(
        out, expected_rmsd, atol=1e-5
    ), "RMSD should be zero for identical trajectories"


def test_different_2d_configurations():
    """Test RMSD for genuinely different 2D configurations (misalignment not recoverable by translation or rotation)."""
    traj1 = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])  # In line along the x-axis
    traj2 = torch.tensor(
        [[[0.0, 1.0], [1.0, 1.0]]]
    )  # In line but offset along the y-axis
    out = rmsd(traj1, traj2)
    print("RMSD:", rmsd)
    assert not torch.allclose(
        out, torch.zeros(1), atol=1e-5
    ), "RMSD should not be zero for misaligned configurations"


def test_rmsd_3d_case():
    traj1 = torch.tensor(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
    )
    traj2 = torch.tensor(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
    )
    expected_rmsd = torch.zeros(2)
    out = rmsd(traj1, traj2)
    assert torch.allclose(
        out, expected_rmsd, atol=1e-5
    ), "RMSD should be zero for identical trajectories"
