import dataclasses
import functools
import operator
from typing import Callable, Dict, TypeVar, Any, Literal

import optree
import torch
from torch import Tensor

from beignet.func.__dataclass import _dataclass
from beignet.func._interact import _force, _safe_sum


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SUZUKI_YOSHIDA_WEIGHTS = {
    1: [
        +1.0000000000000000,
    ],
    3: [
        +0.8289815435887510,
        -0.6579630871775020,
        +0.8289815435887510,
    ],
    5: [
        +0.2967324292201065,
        +0.2967324292201065,
        -0.1869297168804260,
        +0.2967324292201065,
        +0.2967324292201065,
    ],
    7: [
        +0.7845136104775600,
        +0.2355732133593570,
        -1.1776799841788700,
        +1.3151863206839100,
        -1.1776799841788700,
        +0.2355732133593570,
        +0.7845136104775600,
    ],
}

T = TypeVar("T")


class _DispatchByState:
    def __init__(self, fn):
        self._fn = fn

        self._registry = {}

    def __call__(self, state, *args, **kwargs):
        if type(state.positions) in self._registry:
            return self._registry[type(state.positions)](state, *args, **kwargs)

        return self._fn(state, *args, **kwargs)

    def register(self, oftype):
        def register_fn(fn):
            self._registry[oftype] = fn

        return register_fn


@_dataclass
class _Normal:
    mean: Tensor
    var: Tensor

    def sample(self):
        mu, sigma = self.mean, torch.sqrt(self.var)

        return mu + sigma * torch.normal(0.0, 1.0, mu.shape, dtype=mu.dtype)

    def log_prob(self, x):
        return (
            -0.5 * torch.log(2 * torch.pi * self.var)
            - 1 / (2 * self.var) * (x - self.mean) ** 2
        )


@_dataclass
class _NoseHooverChainFunctionsList:
    setup_fn: Callable
    half_step_fn: Callable
    update_mass_fn: Callable


@_dataclass
class _NoseHooverChainState:
    degrees_of_freedom: int = dataclasses.field(metadata={"static": True})
    kinetic_energies: Tensor
    masses: Tensor
    momentums: Tensor
    oscillations: Tensor
    positions: Tensor


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


@_dataclass
class _NVEState:
    positions: Tensor
    momentums: Tensor
    forces: Tensor
    masses: Tensor

    @property
    def velocities(self) -> Tensor:
        return self.momentums / self.masses


@_dataclass
class _NVTLangevinThermostatState:
    forces: Tensor
    masses: Tensor
    momentums: Tensor
    positions: Tensor

    @property
    def velocities(self) -> Tensor:
        return self.momentums / self.masses


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


def _canonicalize_force_fn(fn: Callable[..., Tensor]) -> Callable[..., Tensor]:
    _force_fn = None

    def _fn(_positions: Tensor, **kwargs):
        nonlocal _force_fn

        if _force_fn is None:
            outputs = fn(_positions, **kwargs)

            if outputs.shape == ():
                _force_fn = _force(fn)
            else:

                def _f(x: Tensor, y: Tensor) -> bool:
                    return x.shape == y.shape

                tree_map = optree.tree_map(_f, outputs, _positions)

                def _g(x, y):
                    return x and y

                if not optree.tree_reduce(_g, tree_map, True):
                    raise ValueError

                _force_fn = fn

        return _force_fn(_positions, **kwargs)

    return _fn


@_DispatchByState
def _canonicalize_masses(state: T, **_) -> T:
    def _fn(_mass: float | Tensor) -> float | Tensor:
        if isinstance(_mass, float):
            return _mass

        match _mass.ndim:
            case 0:
                return _mass
            case 1:
                return torch.reshape(_mass, [_mass.shape[0], 1])
            case 2 if _mass.shape[1] == 1:
                return _mass

        raise ValueError

    masses = optree.tree_map(_fn, state.masses)

    return state.set(masses=masses)


def _default_nhc_kwargs(tau: float, overrides: Dict) -> Dict:
    # Copied from https://github.com/jax-md/jax-md/blob/e0ea7d3c235724aac60cecad8dc6f4571d7e5757/jax_md/simulate.py#L485
    default_kwargs = {"size": 3, "steps": 2, "system_steps": 3, "oscillation": tau}

    if overrides is None:
        return default_kwargs

    return {key: overrides.get(key, default_kwargs[key]) for key in default_kwargs}


@functools.singledispatch
def _degrees_of_freedom_metric(positions: Tensor) -> int:
    # util.check_custom_simulation_type(position)

    def _fn(accumulator: Tensor, x: Tensor) -> int:
        return accumulator + torch.numel(x)

    return optree.tree_reduce(_fn, positions, 0)


def _kinetic_energy_metric(
    *,
    momentums: Tensor = None,
    velocities: Tensor = None,
    masses: Tensor = 1.0,
) -> Tensor:
    if momentums is not None and velocities is not None:
        raise ValueError

    if momentums is not None:
        momentums_or_velocities = momentums
    else:
        momentums_or_velocities = velocities

    # _check_custom_simulation_type(q)

    def _kinetic_energy_fn(
        _masses: Tensor,
        _momentums_or_velocities: Tensor,
    ) -> Tensor:
        if momentums is None:

            def k(v, m):
                return v**2 * m
        else:

            def k(p, m):
                return p**2 / m

        return 0.5 * _safe_sum(k(_momentums_or_velocities, _masses))

    kinetic_energy = optree.tree_map(
        _kinetic_energy_fn,
        masses,
        momentums_or_velocities,
    )

    return optree.tree_reduce(operator.add, kinetic_energy, 0.0)


@_DispatchByState
def _momentum_step(state: T, step_size: float) -> T:
    def _fn(_momentums: Tensor, _forces: Tensor) -> Tensor:
        return _momentums + step_size * _forces

    momentums = optree.tree_map(_fn, state.momentums, state.forces)

    return state.set(momentums=momentums)


def nose_hoover_chain(
    step_size: float,
    size: int,
    steps: int,
    system_steps: int,
    oscillation: float,
) -> _NoseHooverChainFunctionsList:
    def setup_fn(
        degrees_of_freedom: Tensor,
        kinetic_energies: Tensor,
        temperature: float,
    ) -> _NoseHooverChainState:
        positions = torch.zeros(size, dtype=kinetic_energies.dtype)

        momentums = torch.zeros(size, dtype=kinetic_energies.dtype)

        masses = torch.ones(size, dtype=torch.float32)

        masses = temperature * oscillation**2.0 * masses

        masses[0] = masses[0] * degrees_of_freedom

        return _NoseHooverChainState(
            degrees_of_freedom=degrees_of_freedom,
            kinetic_energies=kinetic_energies,
            masses=masses,
            momentums=momentums,
            oscillations=oscillation,
            positions=positions,
        )

    def substep_fn(
        time_step_1,
        system_momentums: Tensor,
        state: _NoseHooverChainState,
        temperature: float,
    ) -> (Tensor, _NoseHooverChainState, float):
        (
            degrees_of_freedom,
            kinetic_energies,
            masses,
            momentums,
            _oscillations,
            positions,
        ) = dataclasses.astuple(state)

        time_step_2 = time_step_1 / 2.0
        time_step_4 = time_step_2 / 2.0
        time_step_8 = time_step_4 / 2.0

        m = size - 1

        momentum_correction = momentums[m - 1] ** 2.0 / masses[m - 1] - temperature

        momentums[m] = momentums[m] + (time_step_4 * momentum_correction)

        def backward_fn(updated_momentum: Tensor, m) -> (Tensor, Tensor):
            updated_momentum = torch.multiply(
                torch.exp(
                    torch.divide(
                        torch.multiply(
                            updated_momentum,
                            torch.negative(
                                time_step_8,
                            ),
                        ),
                        masses[m + 1],
                    ),
                ),
                torch.add(
                    torch.multiply(
                        torch.exp(
                            torch.divide(
                                torch.multiply(
                                    updated_momentum,
                                    torch.negative(
                                        time_step_8,
                                    ),
                                ),
                                masses[m + 1],
                            ),
                        ),
                        momentums[m],
                    ),
                    torch.multiply(
                        time_step_4,
                        torch.subtract(
                            torch.divide(
                                torch.square(
                                    momentums[m - 1],
                                ),
                                masses[m - 1],
                            ),
                            temperature,
                        ),
                    ),
                ),
            )

            return updated_momentum, updated_momentum

        indexes = torch.arange(m - 1, 0, -1)

        _, p_xi_update = _scan(
            backward_fn,
            momentums[m],
            indexes,
        )

        momentums[indexes] = p_xi_update

        momentum_correction = 2.0 * kinetic_energies - degrees_of_freedom * temperature

        momentum_scale = torch.exp(-time_step_8 * momentums[1] / masses[1])

        momentums[0] = momentum_scale * (
            momentum_scale * momentums[0] + time_step_4 * momentum_correction
        )

        momentum_scale = torch.exp(-time_step_2 * momentums[0] / masses[0])

        kinetic_energies = kinetic_energies * momentum_scale**2.0

        system_momentums = optree.tree_map(
            lambda p: p * momentum_scale, system_momentums
        )

        positions = positions + time_step_2 * momentums / masses

        momentum_correction = 2.0 * kinetic_energies - degrees_of_freedom * temperature

        def forward_fn(G, m):
            scale = torch.exp(-time_step_8 * momentums[m + 1] / masses[m + 1])

            updated_thermostat_momentum = scale * (
                scale * momentums[m] + time_step_4 * G
            )

            G = updated_thermostat_momentum**2 / masses[m] - temperature

            return G, updated_thermostat_momentum

        indexes = torch.arange(m)

        momentum_correction, p_xi_update = _scan(
            forward_fn, momentum_correction, indexes
        )

        momentums[indexes] = p_xi_update

        momentums[m] = momentums[m] + (torch.multiply(momentum_correction, time_step_4))

        return (
            system_momentums,
            _NoseHooverChainState(
                degrees_of_freedom=degrees_of_freedom,
                kinetic_energies=kinetic_energies,
                masses=masses,
                momentums=momentums,
                oscillations=_oscillations,
                positions=positions,
            ),
            temperature,
        )

    def half_step_fn(system_momentums, state, temperature):
        if steps == 1 and system_steps == 1:
            system_momentums, state, _ = substep_fn(
                step_size,
                system_momentums,
                state,
                temperature,
            )

            return system_momentums, state

        delta = step_size / steps

        weights = torch.tensor(SUZUKI_YOSHIDA_WEIGHTS[system_steps], device=device)

        def body_fn(cs, i):
            d = torch.tensor(delta * weights[i % system_steps], device=device, dtype=torch.float32)
            return substep_fn(d, *cs), 0

        (system_momentums, state, _), _ = _scan(
            body_fn,
            (system_momentums, state, temperature),
            torch.arange(steps * system_steps),
        )

        return system_momentums, state

    def update_mass_fn(
        state: _NoseHooverChainState,
        temperature: float,
    ) -> _NoseHooverChainState:
        (
            degrees_of_freedom,
            kinetic_energies,
            masses,
            momentums,
            oscillations,
            positions,
        ) = dataclasses.astuple(state)

        masses = torch.ones(size, dtype=torch.float32)

        masses = temperature * oscillations**2 * masses

        masses[0] = masses[0] * degrees_of_freedom

        return _NoseHooverChainState(
            degrees_of_freedom=degrees_of_freedom,
            kinetic_energies=kinetic_energies,
            masses=masses,
            momentums=momentums,
            oscillations=oscillations,
            positions=positions,
        )

    return _NoseHooverChainFunctionsList(
        setup_fn=setup_fn,
        half_step_fn=half_step_fn,
        update_mass_fn=update_mass_fn,
    )


def _npt_box_info(
    state: _NPTNoseHooverChainState,
) -> (float, Callable[[float], float]):
    spatial_dimension = state.positions.shape[1]

    reference_box = state.reference_box

    v_0 = _volume_metric(spatial_dimension, reference_box)

    v = torch.multiply(
        v_0,
        torch.exp(
            torch.multiply(
                state.current_box_positions,
                spatial_dimension,
            ),
        ),
    )

    def fn(_v: Tensor) -> Tensor:
        return torch.pow(
            torch.divide(
                _v,
                v_0,
            ),
            torch.multiply(
                reference_box,
                1 / spatial_dimension,
            ),
        )

    return v, fn


def _npt_nose_hoover(
    fn: Callable[..., Tensor],
    shift_fn: Callable[[Tensor, Tensor], Tensor],
    step_size: float,
    pressure: float,
    temperature: float,
    barostat_kwargs: dict | None = None,
    thermostat_kwargs: dict | None = None,
) -> (Callable[..., T], Callable[[T], T]):
    step_size_2 = step_size / 2

    force_fn = _force(fn)

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
            masses = 1.0

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

        current_box_masses = (
            spatial_dimension
            * (particles + 1)
            * _temperature
            * state.barostat.oscillations**2
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
                torch.func.grad(u)(0.0),
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
            masses = 1.0

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


@_DispatchByState
def _positions_step(
    state: T,
    shift_fn: Callable,
    step_size: float,
    **kwargs,
) -> T:
    if isinstance(shift_fn, Callable):

        def _fn(_: Tensor) -> Callable:
            return shift_fn

        shift_fn = optree.tree_map(_fn, state.positions)

    def _fn(
        _shift_fn: Callable,
        _positions: Tensor,
        _momentums: Tensor,
        _masses: Tensor,
    ) -> Tensor:
        return _shift_fn(
            _positions,
            step_size * _momentums / _masses,
            **kwargs,
        )

    positions = optree.tree_map(
        _fn,
        shift_fn,
        state.positions,
        state.momentums,
        state.masses,
    )

    return state.set(positions=positions)


def _scan(fn: Callable, carry: Any, indexes: Tensor):
    """
    Apply a function in a loop sequence over the input index tensor and accumulate the
    intermediate results.

    Parameters
    ----------
    fn : Callable
        The function to be applied iteratively. Takes the current "carry" value and the
        current value from `indexes`, and returns the next "carry" value and an output
        value.
    carry : Any
        The initial "carry" value that gets updated each time `fn` is applied.
    indexes : torch.Tensor
        The 1D index tensor over which to loop.

    Returns
    -------
    carry : Any
        The final "carry" value after applying `fn` to every item in `indexes`.
    ys : torch.Tensor
        Tensor of outputs after each application of `fn`.
    """
    ys = []

    for x in indexes:
        carry, y = fn(carry, x)

        ys.append(y)

    return carry, torch.tensor(ys, device=device)


@_DispatchByState
def _setup_momentum(state: T, temperature: float) -> T:
    positions, masses = state.positions, state.masses

    positions, tree_spec = optree.tree_flatten(positions)

    masses, _ = optree.tree_flatten(masses)

    def _fn(_position: Tensor, _mass: Tensor) -> Tensor:
        _position = _position.to(device=device)
        _mass = _mass.to(device=device)


        sample = torch.normal(
            0.0,
            1.0,
            _position.shape,
            device=_position.device,
            dtype=_position.dtype,
        )

        momentum = torch.sqrt(_mass * temperature) * sample

        if _position.shape[0] > 1:
            momentum = momentum - torch.mean(momentum, dim=0, keepdim=True)

        return momentum

    momentums = []

    for position, mass in zip(positions, masses):
        momentums = [*momentums, _fn(position, mass)]

    momentums = optree.tree_unflatten(tree_spec, momentums)

    return state.set(momentums=momentums)


@_DispatchByState
def _stochastic_step(
    state: _NVTLangevinThermostatState,
    step_size: float,
    temperature: Tensor,
    friction: Tensor,
) -> _NVTLangevinThermostatState:
    c1 = torch.exp(torch.multiply(torch.negative(friction), step_size))

    c2 = torch.sqrt(
        torch.multiply(temperature, torch.subtract(1.0, torch.square(c1)))
    )

    momentum_dist = _Normal(c1 * state.momentums, c2**2 * state.masses)

    return state.set(
        momentums=momentum_dist.sample(),
    )


@_DispatchByState
def _update_kinetic_energy(state: T) -> Tensor:
    return _kinetic_energy_metric(
        masses=state.masses,
        momentums=state.momentums,
    )


def _velocity_verlet(
    force_fn: Callable[..., Tensor],
    shift_fn: Callable[[Tensor, Tensor], Tensor],
    step_size: float,
    state: T,
    **kwargs,
) -> T:
    state = _momentum_step(state, step_size / 2)

    state = _positions_step(state, shift_fn, step_size, **kwargs)

    state = state.set(forces=force_fn(state.positions, **kwargs))

    return _momentum_step(
        state,
        step_size / 2,
    )


def _volume_metric(dimension: int, box: Tensor) -> Tensor:
    if box.shape == torch.Size([]) or not box.ndim:
        return box**dimension

    match box.ndim:
        case 1:
            return torch.prod(box)
        case 2:
            return torch.linalg.det(box)
        case _:
            raise ValueError


def ensemble(
    fn: Callable[..., Tensor],
    shift_fn: Callable[[Tensor, Tensor], Tensor],
    step_size: float,
    *,
    kind: Literal["NPT", "NVE", "NVT"] = "NVE",
    friction: float | Tensor | None = None,
    temperature: float | Tensor | None = None,
    pressure: float | Tensor | None = None,
    barostat: Literal["Nose-Hoover"] | None = None,
    thermostat: Literal["Langevin", "Nose-Hoover"] | None = None,
    barostat_kwargs: dict[str, Any] | None = None,
    thermostat_kwargs: dict[str, Any] | None = None,
) -> (Callable[..., T], Callable[[T], T]):
    r"""

    Parameters
    ----------
    fn : Callable[..., Tensor]
        Function that returns either an energy or a force from
        $\text{positions}$, specified as a PyTorch Tensor of shape
        $(\text{positions} \times \text{spatial_dimension})$.

    shift_fn : Callable[[Tensor, Tensor], Tensor]
        Function that displaces $\text{positions}$ by $\text{distances}$. Both
        $\text{positions}$ and $\text{distances}$ must be PyTorch Tensors of
        shape $(\text{positions} \times \text{spatial_dimension})$.

    step_size : float
        Time step.

    kind : Literal["NPT", "NVE", "NVT"]

    friction : float, optional

    temperature : float, optional
        Temperature in units of Boltzmann constant. You may update the
        temperature dynamically during a simulation by passing `temperature` as
        a keyword argument to the returned step function.

    pressure : float, optional
        Target pressure. You may update the temperature dynamically during a
        simulation by passing `pressure` as a keyword argument to the returned
        step function.

    barostat : Literal["Nose-Hoover"], optional

    thermostat : Literal["Langevin", "Nose-Hoover"], optional

    barostat_kwargs : dict[str, float], optional

    thermostat_kwargs : dict[str, float], optional

    Returns
    -------

    """
    if friction is None:
        friction = 1.0

    if not isinstance(friction, Tensor):
        friction = torch.tensor(friction, device=device)

    if barostat_kwargs is None:
        barostat_kwargs = {}

    if thermostat_kwargs is None:
        thermostat_kwargs = {}

    match kind:
        case "NPT":
            match barostat, thermostat:
                case "Nose-Hoover", "Nose-Hoover":
                    return _npt_nose_hoover(
                        fn=fn,
                        shift_fn=shift_fn,
                        step_size=step_size,
                        pressure=pressure,
                        temperature=temperature,
                        barostat_kwargs=barostat_kwargs,
                        thermostat_kwargs=thermostat_kwargs,
                    )
                case _:
                    raise ValueError
        case "NVE":
            force_fn = _canonicalize_force_fn(fn)

            def setup_fn(
                positions: Tensor,
                temperature: Tensor,
                masses: Tensor | None = None,
                **kwargs,
            ):
                if masses is None:
                    masses = torch.tensor(1.0, device=device, dtype=positions.dtype)

                state = _NVEState(
                    forces=force_fn(positions, **kwargs),
                    masses=masses,
                    momentums=None,
                    positions=positions,
                )

                state = _canonicalize_masses(
                    state=state,
                )

                return _setup_momentum(
                    state=state,
                    temperature=temperature,
                )

            def step_fn(state, **kwargs):
                _step_size = kwargs.pop("step_size", step_size)

                return _velocity_verlet(
                    force_fn=force_fn,
                    shift_fn=shift_fn,
                    step_size=_step_size,
                    state=state,
                    **kwargs,
                )

            return setup_fn, step_fn
        case "NVT":
            if temperature is None:
                raise ValueError

            if not isinstance(temperature, Tensor):
                temperature = torch.tensor(temperature, device=device)

            match thermostat:
                case "Langevin":
                    force_fn = _canonicalize_force_fn(fn)

                    def setup_fn(
                        positions: Tensor,
                        masses: Tensor | None = None,
                        **kwargs,
                    ) -> _NVTLangevinThermostatState:
                        if masses is None:
                            masses = torch.tensor(
                                1.0,
                                device=device,
                                dtype=positions.dtype,
                            )

                        return _setup_momentum(
                            _canonicalize_masses(
                                _NVTLangevinThermostatState(
                                    forces=force_fn(positions, **kwargs),
                                    masses=masses,
                                    momentums=None,
                                    positions=positions,
                                )
                            ),
                            kwargs.pop("temperature", temperature),
                        )

                    def step_fn(
                        state: _NVTLangevinThermostatState,
                        **kwargs,
                    ) -> _NVTLangevinThermostatState:
                        _step_size = kwargs.pop("step_size", step_size)

                        _temperature = kwargs.pop("temperature", temperature)

                        state = _momentum_step(
                            state=state,
                            step_size=_step_size / 2,
                        )

                        state = _positions_step(
                            state=state,
                            shift_fn=shift_fn,
                            step_size=_step_size / 2,
                            **kwargs,
                        )

                        state = _stochastic_step(
                            state=state,
                            step_size=_step_size,
                            temperature=_temperature,
                            friction=friction,
                        )

                        state = _positions_step(
                            state=state,
                            shift_fn=shift_fn,
                            step_size=_step_size / 2,
                            **kwargs,
                        )

                        state = state.set(
                            forces=force_fn(state.positions, **kwargs),
                        )

                        return _momentum_step(
                            state=state,
                            step_size=_step_size / 2,
                        )

                    return setup_fn, step_fn
                case "Nose-Hoover":
                    return _nvt_nose_hoover_chain(
                        fn=fn,
                        shift_fn=shift_fn,
                        step_size=step_size,
                        temperature=temperature,
                        **thermostat_kwargs,
                    )
                case _:
                    raise ValueError
        case _:
            raise ValueError
