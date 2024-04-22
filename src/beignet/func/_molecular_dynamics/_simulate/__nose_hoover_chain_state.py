import dataclasses

from torch import Tensor

from ..__dataclass import _dataclass


@_dataclass
class _NoseHooverChainState:
    degrees_of_freedom: int = dataclasses.field(metadata={"static": True})
    kinetic_energies: Tensor
    masses: Tensor
    momentums: Tensor
    oscillations: Tensor
    positions: Tensor
