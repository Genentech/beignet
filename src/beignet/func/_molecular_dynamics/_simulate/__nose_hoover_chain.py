import dataclasses

import optree
import torch
from torch import Tensor

from .__nose_hoover_chain_functions_list import _NoseHooverChainFunctionsList
from .__nose_hoover_chain_state import _NoseHooverChainState
from .__scan import _scan

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

        weights = torch.tensor(SUZUKI_YOSHIDA_WEIGHTS[system_steps])

        def body_fn(cs, i):
            d = torch.tensor(delta * weights[i % system_steps], dtype=torch.float32)
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
