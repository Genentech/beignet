from typing import TypeVar

from torch import Tensor

from .__dispatch_by_state import _DispatchByState
from .__kinetic_energy_metric import _kinetic_energy_metric

T = TypeVar("T")


@_DispatchByState
def _update_kinetic_energy(state: T) -> Tensor:
    return _kinetic_energy_metric(
        masses=state.masses,
        momentums=state.momentums,
    )
