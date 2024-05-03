import torch
from scipy.spatial.transform import Rotation as R


# TODO (isaacsoh) parallelize and speed up, eliminate 3-D requirement
def _rmsd(traj1, traj2):
    """

    Parameters
    ----------
    traj1 : Trajectory
        For each conformation in this trajectory, compute the RMSD to
        a particular 'reference' conformation in another trajectory.
    traj2 : Trajectory
        The reference conformation to measure distances
        to.

    Returns
    -------
    rmsd_result : torch.Tensor
        The rmsd calculation of two trajectories.
    """

    assert traj1.shape == traj2.shape, "Input tensors must have the same shape"
    assert traj1.dim() == 3, "Input tensors must be 3-D (num_frames, num_atoms, 3)"

    num_frames = traj1.shape[0]  # Number of frames

    # Center the trajectories
    traj1 = traj1 - traj1.mean(dim=1, keepdim=True)
    traj2 = traj2 - traj2.mean(dim=1, keepdim=True)

    # Initialization of the resulting RMSD tensor
    rmsd_result = torch.zeros(num_frames).double()

    for i in range(num_frames):
        # For each configuration compute the rotation matrix minimizing RMSD using SVD
        u, s, v = torch.svd(torch.mm(traj1[i].t(), traj2[i]))

        # Determinat of u * v
        d = (u * v).det().item() < 0.0

        if d:
            s[-1] = s[-1] * (-1)
            u[:, -1] = u[:, -1] * (-1)

        # Optimal rotation matrix
        rot_matrix = torch.mm(v, u.t())

        test = (R.from_matrix(rot_matrix)).as_matrix()

        assert torch.allclose(torch.from_numpy(test), rot_matrix, rtol=1e-03, atol=1e-04)

        # Calculate RMSD and append to resulting tensor
        traj2[i] = torch.mm(traj2[i], rot_matrix)

        rmsd_result[i] = torch.sqrt(
            torch.sum((traj1[i] - traj2[i]) ** 2) / traj1.shape[1])

    return rmsd_result