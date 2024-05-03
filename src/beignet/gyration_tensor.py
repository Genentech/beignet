import torch


def _compute_center_of_geometry(traj):
    """Compute the center of geometry for each frame.

    Parameters
    ----------
    traj : Trajectory
        Trajectory to compute center of geometry for.

    Returns
    -------
    centers : torch.Tensor, shape=(n_frames, 3)
         Coordinates of the center of geometry for each frame.

    """

    centers = torch.zeros((traj.n_frames, 3))

    for i, x in enumerate(traj.xyz):
        centers[i, :] = torch.mean(x.double().t(), dim=1)

    return centers


def gyration_tensor(traj):
    """Compute the gyration tensor of a trajectory.

        For every frame,

        .. math::

            S_{xy} = \sum_{i_atoms} r^{i}_x r^{i}_y

        Parameters
        ----------
        traj : Trajectory
            Trajectory to compute gyration tensor of.

        Returns
        -------
        S_xy:  torch.Tensor, shape=(traj.n_frames, 3, 3), dtype=float64
            Gyration tensors for each frame.

        References
        ----------
        .. [1] https://isg.nist.gov/deepzoomweb/measurement3Ddata_help#shape-metrics-formulas

        """
    center_of_geom = torch.unsqueeze(_compute_center_of_geometry(traj), dim=1)

    xyz = traj.xyz - center_of_geom

    return torch.einsum('...ji,...jk->...ik', xyz, xyz) / traj.n_atoms
