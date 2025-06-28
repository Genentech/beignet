import torch
from torch import Tensor


def dipole_moments(input: Tensor, charges: Tensor) -> Tensor:
    """
    Calculate the dipole moments of each frame in a tensor-represented trajectory using PyTorch.

    Parameters
    ----------
    input : Tensor
        A tensor representing the trajectory with shape (n_frames, n_atoms, 3).
    charges : Tensor
        Charges of each atom in the trajectory. Shape (n_atoms,), units of elementary charges.

    Returns
    -------
    moments : Tensor
        Dipole moments of trajectory, units of nm * elementary charge, shape (n_frames, 3).

    Notes
    -----
    This function performs a straightforward calculation of the dipole moments based on the input atomic positions
    and charges. The dipole moment is calculated as the sum of charge-weighted atomic positions for each frame.
    """
    # Ensure charges is at least 2D: (n_atoms, 1)
    charges = charges.view(-1, 1)

    # Calculate the weighted positions by charges for each frame
    weighted_positions = input * charges

    # TODO (isaacoh) this is broken for symmetric atoms w/ mixed charges
    # Sum over all atoms to get the dipole moment per frame
    moments = torch.sum(weighted_positions, dim=1)

    return moments
