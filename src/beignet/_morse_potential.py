import torch
from torch import Tensor


def morse_potential(
    input: Tensor,
    sigma: Tensor | None = 1.0,
    epsilon: Tensor | None = 5.0,
    alpha: Tensor | None = 5.0,
    **_,
) -> Tensor:
    """
    Morse interaction between particles with a minimum at `sigma`.

    Parameters
    ----------
    input : Tensor, shape=[n, m]
        Pairwise distances between particles.

    sigma : Tensor, optional
        Distance between particles where the energy has a minimum. Should
        either be a floating-point scalar or a Tensor of shape `[n, m]`.

    epsilon : Tensor, optional
        Interaction energy scale. Should either be a floating-point scalar or
        a Tensor of shape `[n, m]`.

    alpha : Tensor, optional
        Range parameter. Should either be a floating-point scalar or a Tensor
        of shape `[n, m]`.

    Returns
    -------
    output : Tensor, shape=[n, m]
        Energies.
    """
    if sigma is None:
        sigma = torch.tensor(1.0, dtype=input.dtype)

    if epsilon is None:
        epsilon = torch.tensor(5.0, dtype=input.dtype)

    if alpha is None:
        alpha = torch.tensor(5.0, dtype=input.dtype)

    return epsilon * (1.0 - torch.exp(-alpha * (input - sigma))) ** 2.0 - epsilon
