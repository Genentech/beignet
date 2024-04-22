from typing import Callable, TypeVar

import torch
from torch import Tensor

from ..__safe_sum import _safe_sum
from .._force import force
from .__canonicalize_masses import _canonicalize_masses
from .__default_nhc_kwargs import _default_nhc_kwargs
from .__degrees_of_freedom_metric import _degrees_of_freedom_metric
from .__kinetic_energy_metric import _kinetic_energy_metric
from .__nose_hoover_chain import nose_hoover_chain
from .__nose_hoover_chain_state import _NoseHooverChainState
from .__npt_box_info import _npt_box_info
from .__npt_nose_hoover_chain_state import _NPTNoseHooverChainState
from .__setup_momentum import _setup_momentum
from .__update_kinetic_energy import _update_kinetic_energy

T = TypeVar("T")


def _npt_nose_hoover(
    fn: Callable[..., Tensor],
    shift_fn: Callable[[Tensor, Tensor], Tensor],
    step_size: float,
    pressure: float,
    temperature: float,
    barostat_kwargs: dict | None = None,
    thermostat_kwargs: dict | None = None,
) -> (Callable[..., T], Callable[[T], T]):
    step_size_2 = torch.tensor(step_size / 2, dtype=torch.float32)

    force_fn = force(fn)

    barostat_kwargs = _default_nhc_kwargs(1000 * step_size, barostat_kwargs)

    barostat = nose_hoover_chain(
        step_size,
        **barostat_kwargs,
    )

    thermostat_kwargs = _default_nhc_kwargs(100 * step_size, thermostat_kwargs)

    thermostat = nose_hoover_chain(
        step_size,
        **thermostat_kwargs,
    )

    def setup_fn(
        positions: Tensor,
        reference_box: Tensor,
        masses: Tensor | None = None,
        **kwargs,
    ):
        if not masses:
            masses = torch.tensor(1.0, dtype=torch.float32)

        particles, spatial_dimension = positions.shape

        if "temperature" not in kwargs:
            _temperature = temperature
        else:
            _temperature = kwargs["temperature"]

        # The box position is defined via pos = (1 / d) log V / V_0.
        zero = torch.zeros((), dtype=positions.dtype)
        one = torch.ones((), dtype=positions.dtype)

        current_box_positions = zero
        current_box_momentums = zero

        current_box_masses = (
            spatial_dimension
            * (particles + 1)
            * temperature
            * barostat_kwargs["oscillation"] ** 2
            * one
        )

        if reference_box.ndim == 0:
            reference_box = torch.eye(positions.shape[-1]) * reference_box

        _barostat = barostat.setup_fn(
            1,
            _kinetic_energy_metric(
                masses=current_box_masses,
                momentums=current_box_momentums,
            ),
            _temperature,
        )

        state = _setup_momentum(
            state=_canonicalize_masses(
                _NPTNoseHooverChainState(
                    barostat=_barostat,
                    current_box_masses=current_box_masses,
                    current_box_momentums=current_box_momentums,
                    current_box_positions=current_box_positions,
                    forces=force_fn(
                        positions,
                        reference_box=reference_box,
                        **kwargs,
                    ),
                    masses=masses,
                    momentums=None,
                    positions=positions,
                    reference_box=reference_box,
                    thermostat=None,
                ),
            ),
            temperature=_temperature,
        )

        kinetic_energies = _update_kinetic_energy(state)

        return state.set(
            thermostat=thermostat.setup_fn(
                _degrees_of_freedom_metric(positions),
                kinetic_energies,
                _temperature,
            )
        )

    def update_box_mass(
        state: _NPTNoseHooverChainState,
        _temperature: Tensor,
    ) -> _NPTNoseHooverChainState:
        particles, spatial_dimension = state.positions.shape

        current_box_masses = torch.tensor(
            spatial_dimension
            * (particles + 1)
            * _temperature
            * state.barostat.oscillations**2,
            dtype=state.positions.dtype,
        )

        return state.set(
            current_box_masses=current_box_masses,
        )

    def box_force(
        alpha,
        vol,
        box_fn,
        position,
        momentum,
        mass,
        force,
        pressure,
        **kwargs,
    ):
        particles, dim = position.shape

        def u(eps):
            return fn(
                position,
                box=box_fn(vol),
                perturbation=(1 + eps).unsqueeze(0),
                **kwargs,
            )

        return torch.subtract(
            torch.subtract(
                torch.multiply(
                    alpha,
                    _safe_sum(
                        torch.divide(
                            torch.square(momentum),
                            mass,
                        ),
                    ),
                ),
                torch.func.grad(u)(torch.tensor(0.0)),
            ),
            torch.multiply(
                torch.multiply(
                    pressure,
                    vol,
                ),
                dim,
            ),
        )

    def sinhx_x(x):
        """Taylor series for sinh(x) / x as x -> 0."""
        return (
            1
            + x**2 / 6
            + x**4 / 120
            + x**6 / 5040
            + x**8 / 362_880
            + x**10 / 39_916_800
        )

    def exp_iL1(box, R, V, V_b, **kwargs):
        x = V_b * step_size
        x_2 = x / 2
        sinhV = sinhx_x(x_2)

        return shift_fn(
            R,
            R * (torch.exp(x) - 1) + step_size * V * torch.exp(x_2) * sinhV,
            box=box,
            **kwargs,
        )

    def exp_i_l2(alpha, momentums, forces, V_b):
        x = alpha * V_b * step_size_2
        return torch.add(
            torch.multiply(
                momentums,
                torch.exp(
                    torch.negative(x),
                ),
            ),
            torch.multiply(
                torch.multiply(
                    torch.multiply(
                        step_size_2,
                        forces,
                    ),
                    sinhx_x(
                        torch.divide(
                            x,
                            2.0,
                        )
                    ),
                ),
                torch.exp(
                    torch.negative(
                        torch.divide(
                            x,
                            2.0,
                        ),
                    ),
                ),
            ),
        )

    def inner_step(
        state: _NPTNoseHooverChainState,
        **kwargs,
    ) -> _NPTNoseHooverChainState:
        _pressure = kwargs.pop("pressure", pressure)

        positions, momentums, masses, forces = (
            state.positions,
            state.momentums,
            state.masses,
            state.forces,
        )

        current_box_positions, current_box_momentums, current_box_masses = (
            state.current_box_positions,
            state.current_box_momentums,
            state.current_box_masses,
        )

        particles, spatial_dimension = positions.shape

        vol, box_fn = _npt_box_info(state)

        alpha = 1 + 1 / particles

        g_e = box_force(
            alpha,
            vol,
            box_fn,
            positions,
            momentums,
            masses,
            forces,
            _pressure,
            **kwargs,
        )

        current_box_momentums = current_box_momentums + step_size_2 * g_e

        momentums = exp_i_l2(
            alpha,
            momentums,
            forces,
            current_box_momentums / current_box_masses,
        )

        current_box_positions = (
            current_box_positions
            + current_box_momentums / current_box_masses * step_size
        )

        state = state.set(
            current_box_positions=current_box_positions,
        )

        vol, box_fn = _npt_box_info(state)

        box = box_fn(vol)

        positions = exp_iL1(
            box,
            positions,
            momentums / masses,
            current_box_momentums / current_box_masses,
        )

        forces = force_fn(positions, box=box, **kwargs)

        momentums = exp_i_l2(
            alpha,
            momentums,
            forces,
            current_box_momentums / current_box_masses,
        )

        g_e = box_force(
            alpha,
            vol,
            box_fn,
            positions,
            momentums,
            masses,
            forces,
            _pressure,
            **kwargs,
        )

        return state.set(
            current_box_masses=current_box_masses,
            current_box_momentums=torch.add(
                current_box_momentums,
                torch.multiply(
                    g_e,
                    step_size_2,
                ),
            ),
            current_box_positions=current_box_positions,
            forces=forces,
            masses=masses,
            momentums=momentums,
            positions=positions,
        )

    def step_fn(
        state: _NPTNoseHooverChainState,
        **kwargs,
    ) -> _NPTNoseHooverChainState:
        _state: _NPTNoseHooverChainState = state

        if "temperature" not in kwargs:
            _temperature = temperature
        else:
            _temperature = kwargs["temperature"]

        _barostat = barostat.update_mass_fn(
            _state.barostat,
            _temperature,
        )

        _thermostat = thermostat.update_mass_fn(
            _state.thermostat,
            _temperature,
        )

        _state = update_box_mass(_state, _temperature)

        current_box_momentums, _barostat = barostat.half_step_fn(
            _state.current_box_momentums,
            _barostat,
            _temperature,
        )

        momentums, _thermostat = thermostat.half_step_fn(
            _state.momentums,
            _thermostat,
            _temperature,
        )

        _state = _state.set(
            current_box_momentums=current_box_momentums,
            momentums=momentums,
        )

        _state = inner_step(_state, **kwargs)

        _barostat: _NoseHooverChainState = _barostat.set(
            kinetic_energies=_kinetic_energy_metric(
                masses=_state.current_box_masses,
                momentums=_state.current_box_momentums,
            ),
        )

        _thermostat = _thermostat.set(
            kinetic_energies=_kinetic_energy_metric(
                masses=_state.masses,
                momentums=_state.momentums,
            ),
        )

        current_box_momentums, _barostat = barostat.half_step_fn(
            _state.current_box_momentums,
            _barostat,
            _temperature,
        )

        momentums, _thermostat = thermostat.half_step_fn(
            _state.momentums,
            _thermostat,
            _temperature,
        )

        return _state.set(
            barostat=_barostat,
            current_box_momentums=current_box_momentums,
            momentums=momentums,
            thermostat=_thermostat,
        )

    return setup_fn, step_fn
