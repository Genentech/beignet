import operator

import optree
from torch import Tensor

from ..__safe_sum import _safe_sum


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
