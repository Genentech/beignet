from typing import Callable, TypeVar

import torch
from torch import Tensor

from .__canonicalize_force_fn import _canonicalize_force_fn
from .__canonicalize_masses import _canonicalize_masses
from .__degrees_of_freedom_metric import _degrees_of_freedom_metric
from .__nose_hoover_chain import nose_hoover_chain
from .__nvt_nose_hoover_chain_state import (
    _NVTNoseHooverChainState,
)
from .__update_kinetic_energy import _update_kinetic_energy
from .__velocity_verlet import _velocity_verlet

T = TypeVar("T")


def _nvt_nose_hoover_chain(
    fn: Callable[..., Tensor],
    shift_fn: Callable[[Tensor, Tensor], Tensor],
    step_size: float,
    temperature: Tensor,
    nose_hoover_thermostat_size: int = 5,
    nose_hoover_thermostat_steps: int = 2,
    system_steps: int = 3,
    oscillation: float | None = None,
    **_,
) -> (Callable[..., T], Callable[[T], T]):
    force_fn = _canonicalize_force_fn(fn)

    if oscillation is None:
        oscillation = step_size * 100

    thermostat_functions = nose_hoover_chain(
        step_size,
        nose_hoover_thermostat_size,
        nose_hoover_thermostat_steps,
        system_steps,
        oscillation,
    )

    def setup_fn(
        positions: Tensor,
        masses: Tensor | None = None,
        **kwargs,
    ) -> _NVTNoseHooverChainState:
        if masses is None:
            masses = torch.tensor(1.0, dtype=positions.dtype)

        if "temperature" not in kwargs:
            _temperature = temperature
        else:
            _temperature = kwargs["temperature"]

        degrees_of_freedom = _degrees_of_freedom_metric(positions)

        state = _NVTNoseHooverChainState(
            positions=positions,
            momentums=None,
            forces=force_fn(positions, **kwargs),
            masses=masses,
            thermostat=None,
        )

        state = _canonicalize_masses(
            state,
            temperature=_temperature,
        )

        return state.set(
            chain=thermostat_functions.setup_fn(
                degrees_of_freedom,
                _update_kinetic_energy(state),
                _temperature,
            ),
        )

    def step_fn(
        state: _NVTNoseHooverChainState,
        **kwargs,
    ) -> _NVTNoseHooverChainState:
        if "temperature" not in kwargs:
            _temperature = temperature
        else:
            _temperature = kwargs["temperature"]

        thermostat = state.thermostat

        thermostat = thermostat_functions.update_mass_fn(
            thermostat,
            _temperature,
        )

        momentums, thermostat = thermostat_functions.half_step_fn(
            system_momentums=state.momentums,
            state=thermostat,
            temperature=_temperature,
        )

        state = state.set(momentums=momentums)

        state: _NVTNoseHooverChainState = _velocity_verlet(
            force_fn,
            shift_fn,
            step_size,
            state,
            **kwargs,
        )

        momentums, thermostat = thermostat_functions.half_step_fn(
            system_momentums=state.momentums,
            state=thermostat.set(
                kinetic_energies=_update_kinetic_energy(state),
            ),
            temperature=_temperature,
        )

        state = state.set(
            momentums=momentums,
            thermostat=thermostat,
        )

        return state

    return setup_fn, step_fn
