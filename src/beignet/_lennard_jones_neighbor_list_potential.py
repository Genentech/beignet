from typing import Optional, Tuple, Callable

import torch
from torch import Tensor

from beignet import lennard_jones_potential
from beignet._multiplicative_isotropic_cutoff import multiplicative_isotropic_cutoff
from beignet.func._interact import interact
from beignet.func._partition import (
    _NeighborListFormat,
    _NeighborList,
    _NeighborListFunctionList,
    neighbor_list,
)
from beignet.func._space import canonicalize_displacement_or_metric
from beignet.func._utils import maybe_downcast


def lennard_jones_neighbor_list_potential(
    displacement_fn: Callable,
    box_size: Tensor,
    kinds: Optional[Tensor] = None,
    sigma: Tensor = torch.tensor(1.0),
    epsilon: Tensor = torch.tensor(1.0),
    alpha: Tensor = torch.tensor(2.0),
    r_onset: Tensor = torch.tensor(2.0),
    r_cutoff: Tensor = torch.tensor(2.5),
    dr_threshold: Tensor = torch.tensor(0.5),
    per_particle: bool = False,
    normalized: bool = False,
    neighbor_list_format: _NeighborListFormat = _NeighborListFormat.ORDERED_SPARSE,
    **neighbor_kwargs,
) -> Tuple[_NeighborListFunctionList, Callable[[Tensor, _NeighborList], Tensor]]:
    r"""Convenience wrapper to compute Lennard-Jones potential using a neighbor list.

    Parameters
    ----------
    displacement_fn : Callable
        A function to compute the displacement between two sets of positions.

    box_size : Tensor
        The size of the simulation box.

    kinds : Tensor, optional
        A tensor specifying the kinds of particles.

    sigma : Tensor, optional
        The distance at which the inter-particle potential is zero. Default is 1.0.

    epsilon : Tensor, optional
        The depth of the potential well. Default is 1.0.

    alpha : Tensor, optional
        A parameter for the potential (not used in the standard Lennard-Jones potential). Default is 2.0.

    r_onset : Tensor, optional
        The distance at which the potential starts to be applied. Default is 2.0.

    r_cutoff : Tensor, optional
        The distance beyond which the potential is not applied. Default is 2.5.

    dr_threshold : Tensor, optional
        The threshold for updating the neighbor list. Default is 0.5.

    per_particle : bool, optional
        Whether to compute the potential per particle. Default is False.

    normalized : bool, optional
        Whether the coordinates are normalized. Default is False.

    neighbor_list_format : _NeighborListFormat, optional
        The format of the neighbor list. Default is _NeighborListFormat.ORDERED_SPARSE.

    **neighbor_kwargs : dict
        Additional keyword arguments for the neighbor list.

    Returns
    -------
    neighbor_fn : _NeighborListFunctionList
        The neighbor list function.

    energy_fn : Callable[[Tensor, _NeighborList], Tensor]
        The energy function to compute the Lennard-Jones potential.
    """
    sigma = maybe_downcast(sigma)
    epsilon = maybe_downcast(epsilon)
    r_onset = maybe_downcast(r_onset) * torch.max(sigma)
    r_cutoff = maybe_downcast(r_cutoff) * torch.max(sigma)
    dr_threshold = maybe_downcast(dr_threshold)

    neighbor_fn = neighbor_list(
        displacement_fn,
        box_size,
        r_cutoff,
        dr_threshold,
        fractional_coordinates=normalized,
        neighbor_list_format=neighbor_list_format,
        **neighbor_kwargs,
    )

    energy_fn = interact(
        multiplicative_isotropic_cutoff(lennard_jones_potential, r_onset, r_cutoff),
        canonicalize_displacement_or_metric(displacement_fn),
        ignore_unused_parameters=True,
        kinds=kinds,
        sigma=sigma,
        epsilon=epsilon,
        dim=(1,) if per_particle else None,
        interaction="neighbor_list",
    )

    return neighbor_fn, energy_fn
