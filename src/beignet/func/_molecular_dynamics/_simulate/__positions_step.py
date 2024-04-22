from typing import Callable, TypeVar

import optree
from torch import Tensor

from .__dispatch_by_state import _DispatchByState

T = TypeVar("T")


@_DispatchByState
def _positions_step(
    state: T,
    shift_fn: Callable,
    step_size: float,
    **kwargs,
) -> T:
    if isinstance(shift_fn, Callable):

        def _fn(_: Tensor) -> Callable:
            return shift_fn

        shift_fn = optree.tree_map(_fn, state.positions)

    def _fn(
        _shift_fn: Callable,
        _positions: Tensor,
        _momentums: Tensor,
        _masses: Tensor,
    ) -> Tensor:
        return _shift_fn(
            _positions,
            step_size * _momentums / _masses,
            **kwargs,
        )

    positions = optree.tree_map(
        _fn,
        shift_fn,
        state.positions,
        state.momentums,
        state.masses,
    )

    return state.set(positions=positions)
