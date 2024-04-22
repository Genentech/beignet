from typing import TypeVar

import optree
import torch
from torch import Tensor

from .__dispatch_by_state import _DispatchByState

T = TypeVar("T")


@_DispatchByState
def _setup_momentum(state: T, temperature: float) -> T:
    positions, masses = state.positions, state.masses

    positions, tree_spec = optree.tree_flatten(positions)

    masses, _ = optree.tree_flatten(masses)

    def _fn(_position: Tensor, _mass: Tensor) -> Tensor:
        sample = torch.normal(
            0.0,
            1.0,
            _position.shape,
            device=_position.device,
            dtype=_position.dtype,
        )

        momentum = torch.sqrt(_mass * temperature) * sample

        if _position.shape[0] > 1:
            momentum = momentum - torch.mean(momentum, dim=0, keepdim=True)

        return momentum

    momentums = []

    for position, mass in zip(positions, masses):
        momentums = [*momentums, _fn(position, mass)]

    momentums = optree.tree_unflatten(tree_spec, momentums)

    return state.set(momentums=momentums)
