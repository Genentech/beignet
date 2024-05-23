import torch
from torch import Tensor


def bond_angles(input: Tensor, angle_indices: Tensor) -> Tensor:
    """
    Compute the bond angles between the supplied triplets of indices in each frame of a trajectory using PyTorch.

    Parameters
    ----------
    input : Tensor
        Trajectory tensor with shape=(n_frames, n_atoms, 3).
    angle_indices : Tensor
        Tensor of shape=(num_angles, 3), each row consists of indices of three atoms.

    Returns
    -------
    angles : Tensor
        Angles for each specified group of indices, shape=(n_frames, num_angles). Angles are in radians.
    """
    # Data verification
    num_frames, n_atoms, _ = input.shape
    if torch.any(angle_indices >= n_atoms) or torch.any(angle_indices < 0):
        raise ValueError("angle_indices must be between 0 and %d" % (n_atoms - 1))

    if angle_indices.shape[0] == 0:
        return torch.zeros((num_frames, 0), dtype=torch.float32)

    # Initializing the output tensor
    angles = torch.zeros((num_frames, angle_indices.shape[0]), dtype=torch.float32)

    # Gathering vectors related to the angle calculation
    vec1 = input[:, angle_indices[:, 1]] - input[:, angle_indices[:, 0]]
    vec2 = input[:, angle_indices[:, 1]] - input[:, angle_indices[:, 2]]

    # Normalize the vectors
    vec1_norm = torch.norm(vec1, dim=2, keepdim=True)
    vec2_norm = torch.norm(vec2, dim=2, keepdim=True)
    vec1_unit = vec1 / vec1_norm
    vec2_unit = vec2 / vec2_norm

    # Compute angles using arccos of dot products
    dot_products = torch.sum(vec1_unit * vec2_unit, dim=2)
    angles = torch.acos(dot_products)

    return angles
