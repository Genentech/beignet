from torch import Tensor

from ..__dataclass import _dataclass


@_dataclass
class _NVEState:
    positions: Tensor
    momentums: Tensor
    forces: Tensor
    masses: Tensor

    @property
    def velocities(self) -> Tensor:
        return self.momentums / self.masses
