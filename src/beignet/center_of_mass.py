from torch import Tensor


def center_of_mass(input: Tensor, masses: Tensor) -> Tensor:
    r"""Compute the center of mass for each frame.

    Parameters
    ----------
    input : Tensor
        A tensor of shape (n_frames, n_atoms, 3) which contains the XYZ coordinates of atoms in each frame.
    masses : Tensor
        A tensor of shape (n_atoms,) containing the masses of each atom.

    Returns
    -------
    output : Tensor, shape=(n_frames, 3)
         A tensor of shape (n_frames, 3) with the coordinates of the center of mass for each frame.
    """
    total_mass = masses.sum()

    weighted_positions = input * masses[:, None]

    return weighted_positions.sum(dim=1) / total_mass
