import math

import torch
from torch import Tensor


def dihedrals(input: Tensor, indices: Tensor) -> Tensor:
    r"""
    Compute the dihedral angles between specified quartets of atoms in each frame of a trajectory using PyTorch.

    Parameters
    ----------
    input : Tensor
        A tensor representing the trajectory with shape (n_frames, n_atoms, 3).
    indices : Tensor
        Each row gives the indices of four atoms which, together, define a dihedral angle,
        shape (n_dihedrals, 4).

    Returns
    -------
    Tensor
        A tensor of dihedral angles in radians, shape = (n_frames, n_dihedrals).
    """
    n_frames, n_atoms, _ = input.shape
    if torch.any(indices >= n_atoms) or torch.any(indices < 0):
        raise ValueError("indices must be between 0 and %d" % (n_atoms - 1))

    if len(indices) == 0:
        return torch.zeros((n_frames, 0), dtype=torch.float32)

    # Get vectors between atoms
    vec1 = input[:, indices[:, 1]] - input[:, indices[:, 0]]
    vec2 = input[:, indices[:, 2]] - input[:, indices[:, 1]]
    vec3 = input[:, indices[:, 3]] - input[:, indices[:, 2]]

    # Compute normals to the planes defined by the first three and last three atoms
    normal1 = torch.cross(vec1, vec2, dim=2)
    normal2 = torch.cross(vec2, vec3, dim=2)

    # Compute norms and check for zero to avoid division by zero
    norm1 = torch.norm(normal1, dim=2, keepdim=True)
    norm2 = torch.norm(normal2, dim=2, keepdim=True)
    normal1 = torch.where(norm1 > 0, normal1 / norm1, normal1)
    normal2 = torch.where(norm2 > 0, normal2 / norm2, normal2)

    cosine = torch.sum(normal1 * normal2, dim=2)
    cosine = torch.clamp(cosine, -1.0, 1.0)

    cross = torch.cross(normal1, normal2, dim=2)
    sine = torch.norm(cross, dim=2) * torch.sign(torch.sum(cross * vec2, dim=2))

    # Handle case where the cross product is zero - indicating collinear points
    angle = torch.atan2(sine, cosine)
    angle = torch.where((norm1 > 0) & (norm2 > 0), angle, torch.tensor(math.pi))

    return angle
