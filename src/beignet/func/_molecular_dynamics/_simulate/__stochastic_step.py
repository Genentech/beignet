import torch
from torch import Tensor

from .__dispatch_by_state import _DispatchByState
from .__normal import _Normal
from .__nvt_langevin_thermostat_state import _NVTLangevinThermostatState


@_DispatchByState
def _stochastic_step(
    state: _NVTLangevinThermostatState,
    step_size: float,
    temperature: Tensor,
    friction: Tensor,
) -> _NVTLangevinThermostatState:
    c1 = torch.exp(torch.multiply(torch.negative(friction), step_size))

    c2 = torch.sqrt(
        torch.multiply(temperature, torch.subtract(torch.tensor(1.0), torch.square(c1)))
    )

    momentum_dist = _Normal(c1 * state.momentums, c2**2 * state.masses)

    return state.set(
        momentums=momentum_dist.sample(),
    )
