import torch


def compute_center_of_mass(traj):
    """Compute the center of mass for each frame.

    Parameters
    ----------
    traj : Trajectory
        Trajectory to compute center of mass for

    Returns
    -------
    com : torch.Tensor, shape=(n_frames, 3)
         Coordinates of the center of mass for each frame
    """

    com = torch.empty((traj.n_frames, 3))

    masses = torch.tensor([a.element.mass for a in traj.top.atoms])
    masses /= masses.sum()

    xyz = traj.xyz

    for i, x in enumerate(xyz):
        com[i, :] = torch.tensordot(masses, x.double().t(), dims=0)

    return com
