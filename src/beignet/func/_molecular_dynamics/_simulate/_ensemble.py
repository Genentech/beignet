from typing import Any, Callable, Literal, TypeVar

import torch
from torch import Tensor

from .__canonicalize_force_fn import _canonicalize_force_fn
from .__canonicalize_masses import _canonicalize_masses
from .__momentum_step import _momentum_step
from .__npt_nose_hoover import _npt_nose_hoover
from .__nve_state import _NVEState
from .__nvt_langevin_thermostat_state import _NVTLangevinThermostatState
from .__nvt_nose_hoover_chain import _nvt_nose_hoover_chain
from .__positions_step import _positions_step
from .__setup_momentum import _setup_momentum
from .__stochastic_step import _stochastic_step
from .__velocity_verlet import _velocity_verlet

T = TypeVar("T")


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
        friction = torch.tensor(1.0)

    if not isinstance(friction, Tensor):
        friction = torch.tensor(friction)

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
                    masses = torch.tensor(1.0, dtype=positions.dtype)

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
                temperature = torch.tensor(temperature)

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
