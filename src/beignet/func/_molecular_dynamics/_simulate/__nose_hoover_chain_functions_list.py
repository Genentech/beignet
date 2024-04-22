from typing import Callable

from ..__dataclass import _dataclass


@_dataclass
class _NoseHooverChainFunctionsList:
    setup_fn: Callable
    half_step_fn: Callable
    update_mass_fn: Callable
