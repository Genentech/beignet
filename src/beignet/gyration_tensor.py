import torch
from torch import Tensor


def _compute_center_of_geometry(input: Tensor) -> Tensor:
    """Compute the center of geometry for each frame.
    Parameters
    ----------
    input : Tensor
        Trajectory to compute center of geometry for, shape=(n_frames, n_atoms, 3)

    Returns
    -------
    centers : Tensor, shape=(n_frames, 3)
         Coordinates of the center of geometry for each frame.
    """
    centers = torch.mean(input, dim=1)
    return centers


def gyration_tensor(input: Tensor) -> Tensor:
    """Compute the gyration tensor of a trajectory.

    Parameters
    ----------
    input : Tensor
        Trajectory for which to compute gyration tensor, shape=(n_frames, n_atoms, 3)

    Returns
    -------
    gyration_tensors: Tensor, shape=(n_frames, 3, 3)
        Gyration tensors for each frame.

    References
    ----------
    .. [1] https://isg.nist.gov/deepzoomweb/measurement3Ddata_help#shape-metrics-formulas
    """
    n_frames, n_atoms, _ = input.shape
    center_of_geometry = _compute_center_of_geometry(input).unsqueeze(1)

    # Translate the atoms by subtracting the center of geometry
    translated_trajectory = input - center_of_geometry

    # Compute gyration tensor for each frame
    gyration_tensors = (
        torch.einsum("nij,nik->njk", translated_trajectory, translated_trajectory)
        / n_atoms
    )

    return gyration_tensors
