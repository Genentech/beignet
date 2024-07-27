from typing import Callable, Optional

import torch
from torch import Tensor

from beignet import lennard_jones_potential
from beignet._multiplicative_isotropic_cutoff import multiplicative_isotropic_cutoff
from beignet.func._interact import interact
from beignet.func._space import canonicalize_displacement_or_metric
from beignet.func._utils import maybe_downcast


def lennard_jones_pair_potential(
    displacement_fn: Callable,
    kinds: Optional[Tensor] = None,
    sigma: Tensor = torch.tensor(1.0),
    epsilon: Tensor = torch.tensor(1.0),
    r_onset: Tensor = torch.tensor(2.0),
    r_cutoff: Tensor = torch.tensor(2.5),
    per_particle: bool = False,
) -> Callable[[Tensor], Tensor]:
    r"""Convenience wrapper to compute Lennard-Jones energy over a system.

    Parameters
    ----------
    displacement_fn : Callable
        A function to compute the displacement between two sets of positions.

    kinds : Tensor, optional
        A tensor specifying the kinds of particles.

    sigma : Tensor, optional
        The distance at which the inter-particle potential is zero. Default is 1.0.

    epsilon : Tensor, optional
        The depth of the potential well. Default is 1.0.

    r_onset : Tensor, optional
        The distance at which the potential starts to be applied. Default is 2.0.

    r_cutoff : Tensor, optional
        The distance beyond which the potential is not applied. Default is 2.5.

    per_particle : bool, optional
        Whether to compute the potential per particle. Default is False.

    Returns
    -------
    energy_fn : Callable[[Tensor], Tensor]
        A function that computes the Lennard-Jones potential for a given set of positions.
    """
    sigma = maybe_downcast(sigma)
    epsilon = maybe_downcast(epsilon)
    r_onset = maybe_downcast(r_onset) * torch.max(sigma)
    r_cutoff = maybe_downcast(r_cutoff) * torch.max(sigma)

    return interact(
        multiplicative_isotropic_cutoff(lennard_jones_potential, r_onset, r_cutoff),
        canonicalize_displacement_or_metric(displacement_fn),
        ignore_unused_parameters=True,
        kinds=kinds,
        sigma=sigma,
        epsilon=epsilon,
        dim=(1,) if per_particle else None,
        interaction="pair",
    )
