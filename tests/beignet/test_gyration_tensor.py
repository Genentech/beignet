import torch

from beignet.gyration_tensor import gyration_tensor, _compute_center_of_geometry


def test_center_of_geometry_origin():
    # Scenario where all atoms are at the origin
    traj = torch.zeros(1, 10, 3)  # 1 frame, 10 atoms, all at origin
    expected_center = torch.zeros(1, 3)
    computed_center = _compute_center_of_geometry(traj)
    assert torch.allclose(
        computed_center, expected_center
    ), "Center of geometry should be at origin for zeroed data."


def test_gyration_tensor_origin():
    # All atoms at the origin, expecting zero gyration tensor
    traj = torch.zeros(1, 10, 3)
    expected_gyration = torch.zeros(1, 3, 3)
    computed_gyration = gyration_tensor(traj)
    assert torch.allclose(
        computed_gyration, expected_gyration
    ), "Gyration tensor should be zero for zeroed data."


def test_center_of_geometry_translation():
    # Translate structure along x-axis
    traj = torch.zeros(1, 10, 3) + torch.tensor([[[5.0, 0.0, 0.0]]])
    expected_center = torch.tensor([[5.0, 0.0, 0.0]])
    computed_center = _compute_center_of_geometry(traj)
    assert torch.allclose(
        computed_center, expected_center
    ), "Translation mismatch in calculated center of geometry."


def test_gyration_tensor_translation():
    # Gyration tensor for translated atoms that are spaced uniformly along x from 1 to 10
    traj = torch.arange(1, 11).float().view(1, 10, 1).expand(-1, -1, 3)
    centers = _compute_center_of_geometry(traj)

    deviations = traj[:, :, 0] - centers[:, 0].unsqueeze(1)
    expected_s_xx = torch.sum(deviations**2, dim=1) / traj.shape[1]

    computed_gyration = gyration_tensor(traj)

    assert torch.allclose(
        computed_gyration[:, 0, 0], expected_s_xx
    ), "Gyration tensor calculation error after translation."


def test_random_data():
    # Check with random data to ensure no errors occur in general usage
    traj = torch.randn(5, 50, 3)
    assert gyration_tensor(traj).shape == (
        5,
        3,
        3,
    ), "Unexpected output shape for gyration tensor with random data."
