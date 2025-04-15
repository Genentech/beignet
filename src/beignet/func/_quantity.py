from typing import Optional, Union, Any, Callable

import optree
import torch
from torch import Tensor

from beignet.func._interact import _safe_sum


CUSTOM_SIMULATION_TYPE = []


def check_custom_simulation_type(x: Any) -> bool:
    if type(x) in CUSTOM_SIMULATION_TYPE:
        raise ValueError()


def count_dof(position: Union[Tensor, list, tuple]) -> int:
    """
    Counts the degrees of freedom (DOF) in the given position tensor.

    Parameters
    ----------
    position : Union[Tensor, list, tuple]
        The position tensor or a nested structure of tensors.

    Returns
    -------
    int
        The total number of degrees of freedom.
    """
    check_custom_simulation_type(position)

    return optree.tree_reduce(lambda accum, x: accum + x.numel(), position, 0)


def kinetic_energy(
    *unused_args,
    momentum: Optional[Tensor] = None,
    velocity: Optional[Tensor] = None,
    mass: Union[float, Tensor] = 1.0,
) -> float:
    """
    Computes the kinetic energy of a system.

    To avoid ambiguity, either momentum or velocity must be passed explicitly
    as a keyword argument.

    Parameters
    ----------
    momentum : Optional[Tensor], optional
        Tensor specifying the momentum of the system.
    velocity : Optional[Tensor], optional
        Tensor specifying the velocity of the system.
    mass : Union[float, Tensor], optional
        Tensor specifying the mass of the constituents, by default 1.0.

    Returns
    -------
    float
        The kinetic energy of the system.

    Raises
    ------
    ValueError
        If both momentum and velocity are provided or if neither is provided.
    """
    if unused_args:
        raise ValueError(
            "To use the kinetic energy function, you must explicitly "
            "pass either momentum or velocity as a keyword argument."
        )
    if momentum is not None and velocity is not None:
        raise ValueError(
            "To use the kinetic energy function, you must pass either"
            " a momentum or a velocity."
        )

    k = (lambda v, m: v**2 * m) if momentum is None else (lambda p, m: p**2 / m)
    q = velocity if momentum is None else momentum

    ke = optree.tree_map(lambda m, q: 0.5 * _safe_sum(k(q, m)), mass, q)
    return optree.tree_reduce(lambda x, y: x + y, ke, 0.0).item()


def temperature(
    *unused_args,
    momentum: Optional[Tensor] = None,
    velocity: Optional[Tensor] = None,
    mass: Union[float, Tensor] = 1.0,
) -> float:
    """
    Computes the temperature of a system.

    To avoid ambiguity, either momentum or velocity must be passed explicitly
    as a keyword argument.

    Parameters
    ----------
    momentum : Optional[Tensor], optional
        Tensor specifying the momentum of the system.
    velocity : Optional[Tensor], optional
        Tensor specifying the velocity of the system.
    mass : Union[float, Tensor], optional
        Tensor specifying the mass of the constituents, by default 1.0.

    Returns
    -------
    float
        The temperature of the system in units of the Boltzmann constant.

    Raises
    ------
    ValueError
        If both momentum and velocity are provided or if neither is provided.
    """
    if unused_args:
        raise ValueError(
            "To use the temperature function, you must explicitly "
            "pass either momentum or velocity as a keyword argument."
        )
    if momentum is not None and velocity is not None:
        raise ValueError(
            "To use the temperature function, you must pass either"
            " a momentum or a velocity."
        )

    t = (lambda v, m: v**2 * m) if momentum is None else (lambda p, m: p**2 / m)
    q = velocity if momentum is None else momentum

    def _safe_sum(tensor):
        # Placeholder for the actual implementation of safe sum.
        return tensor.sum()

    dof = count_dof(q)

    kT = optree.tree_map(lambda m, q: _safe_sum(t(q, m)) / dof, mass, q)
    return optree.tree_reduce(lambda x, y: x + y, kT, 0.0).item()


def box_size_at_number_density(
    particle_count: int, number_density: float, spatial_dimension: int
) -> float:
    """
    Computes the box size given the number of particles, number density, and spatial dimension.

    Parameters
    ----------
    particle_count : int
        The number of particles.
    number_density : float
        The number density of the particles.
    spatial_dimension : int
        The spatial dimension of the system.

    Returns
    -------
    float
        The computed box size.
    """
    return torch.pow(particle_count / number_density, 1 / spatial_dimension).item()


def volume(dimension: int, box: Union[float, Tensor]) -> Tensor:
    """
    Computes the volume of a box given its dimensions.

    Parameters
    ----------
    dimension : int
        The dimension of the space.
    box : Union[float, Tensor]
        The box, which can be a scalar, a vector, or a matrix.

    Returns
    -------
    Tensor
        The volume of the box.

    Raises
    ------
    ValueError
        If the box is not a scalar, vector, or matrix.
    """
    if isinstance(box, (int, float)) or not box.ndim:
        return torch.tensor(box**dimension)
    elif box.ndim == 1:
        return torch.prod(box)
    elif box.ndim == 2:
        return torch.det(box)
    else:
        raise ValueError(
            ("Box must be either: a scalar, a vector, or a matrix. " f"Found {box}.")
        )


def pressure(
    energy_fn: Callable,
    position: Tensor,
    box: Union[float, Tensor],
    kinetic_energy: float = 0.0,
    **kwargs,
) -> float:
    """
    Computes the internal pressure of a system.

    Parameters
    ----------
    energy_fn : Callable
        A function that computes the energy of the system. This function must take
        as an argument `perturbation` which perturbs the box shape.
    position : Tensor
        An array of particle positions.
    box : Union[float, Tensor]
        A box specifying the shape of the simulation volume. Used to infer the
        volume of the unit cell.
    kinetic_energy : float, optional
        A float specifying the kinetic energy of the system, by default 0.0.

    Returns
    -------
    float
        A float specifying the pressure of the system.
    """
    dim = position.shape[1]

    def U(eps):
        try:
            return energy_fn(position, box=box, perturbation=(1 + eps), **kwargs)
        except Exception:  # Replace with specific exception if known
            return energy_fn(position, perturbation=(1 + eps), **kwargs)

    def grad_U(eps):
        eps_tensor = torch.tensor([eps], requires_grad=True)
        energy = U(eps_tensor)
        grad_eps = torch.autograd.grad(energy, eps_tensor, create_graph=True)[0]
        return grad_eps

    vol_0 = volume(dim, box)

    return (1 / (dim * vol_0)) * (2 * kinetic_energy - grad_U(0.0).item())
