import torch
from torch import Tensor

from ..__dataclass import _dataclass
from .__nose_hoover_chain_state import _NoseHooverChainState
from .__volume_metric import _volume_metric


@_dataclass
class _NPTNoseHooverChainState:
    positions: Tensor
    momentums: Tensor
    forces: Tensor
    masses: Tensor

    reference_box: Tensor

    current_box_positions: Tensor
    current_box_momentums: Tensor
    current_box_masses: Tensor

    barostat: _NoseHooverChainState
    thermostat: _NoseHooverChainState

    @property
    def current_box(self) -> Tensor:
        return torch.multiply(
            torch.pow(
                torch.divide(
                    torch.multiply(
                        _volume_metric(
                            self.positions.shape[1],
                            self.reference_box,
                        ),
                        torch.exp(
                            torch.multiply(
                                self.current_box_positions,
                                self.positions.shape[1],
                            ),
                        ),
                    ),
                    _volume_metric(
                        self.positions.shape[1],
                        self.reference_box,
                    ),
                ),
                1 / self.positions.shape[1],
            ),
            self.reference_box,
        )

    @property
    def velocities(self) -> Tensor:
        return torch.divide(
            self.momentums,
            self.masses,
        )
