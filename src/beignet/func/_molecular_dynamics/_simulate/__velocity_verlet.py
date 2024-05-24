from typing import Callable, TypeVar

from torch import Tensor

from .__momentum_step import _momentum_step
from .__positions_step import _positions_step

T = TypeVar("T")


def _velocity_verlet(
    force_fn: Callable[..., Tensor],
    shift_fn: Callable[[Tensor, Tensor], Tensor],
    step_size: float,
    state: T,
    **kwargs,
) -> T:
    state = _momentum_step(state, step_size / 2)

    state = _positions_step(state, shift_fn, step_size, **kwargs)

    state = state.set(forces=force_fn(state.positions, **kwargs))

    return _momentum_step(
        state,
        step_size / 2,
    )
