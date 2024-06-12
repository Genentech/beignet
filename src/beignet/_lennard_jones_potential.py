from torch import Tensor


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
    a = sigma / input

    b = a**6.0
    c = b**2.0

    return 4.0 * epsilon * (c - b)
