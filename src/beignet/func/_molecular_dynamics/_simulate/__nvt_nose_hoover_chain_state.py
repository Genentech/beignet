from torch import Tensor

from ..__dataclass import _dataclass
from .__nose_hoover_chain_state import _NoseHooverChainState


@_dataclass
class _NVTNoseHooverChainState:
    positions: Tensor
    momentums: Tensor
    forces: Tensor
    masses: Tensor
    thermostat: _NoseHooverChainState

    @property
    def velocities(self):
        return self.momentums / self.masses
