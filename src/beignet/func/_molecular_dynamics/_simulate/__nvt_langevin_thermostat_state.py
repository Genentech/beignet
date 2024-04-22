from torch import Tensor

from ..__dataclass import _dataclass


@_dataclass
class _NVTLangevinThermostatState:
    forces: Tensor
    masses: Tensor
    momentums: Tensor
    positions: Tensor

    @property
    def velocities(self) -> Tensor:
        return self.momentums / self.masses
