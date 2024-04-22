import functools
from typing import Callable, Optional

import torch
from torch import Tensor

from ..__safe_sum import _safe_sum
from .__merge_dictionaries import _merge_dictionaries
from .__to_bond_kind_parameters import _to_bond_kind_parameters


def _bond_interaction(
    fn: Callable[..., Tensor],
    displacement_fn: Callable[[Tensor, Tensor], Tensor],
    static_bonds: Optional[Tensor] = None,
    static_kinds: Optional[Tensor] = None,
    ignore_unused_parameters: bool = False,
    **static_kwargs,
) -> Callable[..., Tensor]:
    merge_dictionaries_fn = functools.partial(
        _merge_dictionaries,
        ignore_unused_parameters=ignore_unused_parameters,
    )

    def mapped_fn(
        positions: Tensor,
        bonds: Optional[Tensor] = None,
        kinds: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        accumulator = torch.tensor(
            0.0,
            device=positions.device,
            dtype=positions.dtype,
        )

        distance_fn = functools.partial(displacement_fn, **kwargs)

        distance_fn = torch.func.vmap(distance_fn, 0, 0)

        if bonds is not None:
            parameters = merge_dictionaries_fn(static_kwargs, kwargs)

            for name, parameter in parameters.items():
                if kinds is not None:
                    parameters[name] = _to_bond_kind_parameters(
                        parameter,
                        kinds,
                    )

            interactions = distance_fn(
                positions[bonds[:, 0]],
                positions[bonds[:, 1]],
            )

            interactions = _safe_sum(fn(interactions, **parameters))

            accumulator = accumulator + interactions

        if static_bonds is not None:
            parameters = merge_dictionaries_fn(static_kwargs, kwargs)

            for name, parameter in parameters.items():
                if static_kinds is not None:
                    parameters[name] = _to_bond_kind_parameters(
                        parameter,
                        static_kinds,
                    )

            interactions = distance_fn(
                positions[static_bonds[:, 0]],
                positions[static_bonds[:, 1]],
            )

            interactions = _safe_sum(fn(interactions, **parameters))

            accumulator = accumulator + interactions

        return accumulator

    return mapped_fn
