from typing import TypeVar

import optree
import torch
from torch import Tensor

from .__dispatch_by_state import _DispatchByState

T = TypeVar("T")


@_DispatchByState
def _canonicalize_masses(state: T, **_) -> T:
    def _fn(_mass: float | Tensor) -> float | Tensor:
        if isinstance(_mass, float):
            return _mass

        match _mass.ndim:
            case 0:
                return _mass
            case 1:
                return torch.reshape(_mass, [_mass.shape[0], 1])
            case 2 if _mass.shape[1] == 1:
                return _mass

        raise ValueError

    masses = optree.tree_map(_fn, state.masses)

    return state.set(masses=masses)
