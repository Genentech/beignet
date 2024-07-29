import torch
from torch import Tensor


def soft_sphere_potential(input: Tensor,
                sigma: Tensor = 1,
                epsilon: Tensor = 1,
                alpha: Tensor = 2,
                **unused_kwargs) -> Tensor:
    r"""
    Finite ranged repulsive interaction between soft spheres.

    Parameters
    ----------
    input : Tensor
        A tensor of shape `[n, m]` of pairwise distances between particles.
    sigma : Tensor, optional
        Particle diameter. Should either be a floating point scalar or a tensor
        whose shape is `[n, m]`. Default is 1.
    epsilon : Tensor, optional
        Interaction energy scale. Should either be a floating point scalar or a tensor
        whose shape is `[n, m]`. Default is 1.
    alpha : Tensor, optional
        Exponent specifying interaction stiffness. Should either be a floating point scalar
        or a tensor whose shape is `[n, m]`. Default is 2.
    unused_kwargs : dict, optional
        Allows extra data (e.g. time) to be passed to the energy.

    Returns
    -------
    Tensor
        Matrix of energies whose shape is `[n, m]`.
    """
    input = input / sigma
    fn = lambda dr: epsilon / alpha * (1.0 - dr) ** alpha

    if isinstance(alpha, int) or issubclass(type(alpha.dtype), torch.int):
        return torch.where(input < 1.0, fn(input), torch.tensor(0.0, dtype=input.dtype))

    return torch.where(input < 1.0, fn(input), torch.tensor(0.0, dtype=input.dtype))
