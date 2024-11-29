import torch
from torch import Tensor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def lennard_jones_potential(
    input: Tensor,
    sigma: float | Tensor,
    epsilon: float | Tensor,
) -> Tensor:
    r"""
    Lennard-Jones potential.

    Parameters
    ----------
    input : Tensor, shape=(n, m)
        Pairwise distances between particles.

    sigma : float | Tensor, shape=(n, m)
        Distance where the potential energy, :math:`V`, is zero.

    epsilon : float | Tensor, shape=(n, m)
        Depth of the potential well.

    Returns
    -------
    output : Tensor, shape=(n, m)
        Energies.
    """

    a = sigma.to(device=device) / input.to(device=device)

    b = a**6.0
    c = b**2.0

    return torch.nan_to_num(4.0 * epsilon * (c - b))
