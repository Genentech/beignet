from typing import TypeVar

import optree
from torch import Tensor

from .__dispatch_by_state import _DispatchByState

T = TypeVar("T")


@_DispatchByState
def _momentum_step(state: T, step_size: float) -> T:
    def _fn(_momentums: Tensor, _forces: Tensor) -> Tensor:
        return _momentums + step_size * _forces

    momentums = optree.tree_map(_fn, state.momentums, state.forces)

    return state.set(momentums=momentums)
