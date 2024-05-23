from torch import Tensor
import torch


# TODO (isaacsoh) parallelize and speed up, eliminate 3-D requirement
def rmsd(traj1: Tensor, traj2: Tensor):
    """
    Compute the Root Mean Square Deviation (RMSD) between two trajectories.

    Parameters
    ----------
    traj1 : Tensor
        First trajectory tensor, shape (num_frames, num_atoms, dim).
    traj2 : Tensor
        Second trajectory tensor (reference), same shape as traj1.

    Returns
    -------
    rmsd_result : Tensor
        The RMSD calculation of two trajectories.
    """
    assert traj1.shape == traj2.shape, "Input tensors must have the same shape"

    num_frames = traj1.shape[0]
    rmsd_result = torch.zeros(num_frames)

    for i in range(num_frames):
        traj1_centered = traj1[i] - traj1[i].mean(dim=0, keepdim=True)
        traj2_centered = traj2[i] - traj2[i].mean(dim=0, keepdim=True)

        u, s, vh = torch.linalg.svd(torch.mm(traj1_centered.t(), traj2_centered))
        d = torch.sign(torch.det(torch.mm(vh.t(), u.t())))

        if d < 0:
            vh[:, -1] *= -1

        rot_matrix = torch.mm(vh.t(), u.t())
        traj2_rotated = torch.mm(traj2_centered, rot_matrix)

        rmsd = torch.sqrt(((traj1_centered - traj2_rotated) ** 2).sum(dim=1).mean())

        rmsd_result[i] = rmsd

    return rmsd_result
