import functools
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from ..__safe_sum import _safe_sum
from .._partition.__is_neighbor_list_sparse import _is_neighbor_list_sparse
from .._partition.__map_bond import _map_bond
from .._partition.__map_neighbor import _map_neighbor
from .._partition.__neighbor_list import _NeighborList
from .._partition.__neighbor_list_format import _NeighborListFormat
from .._partition.__segment_sum import _segment_sum
from .__kwargs_to_neighbor_list_parameters import (
    _kwargs_to_neighbor_list_parameters,
)
from .__merge_dictionaries import _merge_dictionaries


def _neighbor_list_interaction(
    fn: Callable[..., Tensor],
    displacement_fn: Callable[[Tensor, Tensor], Tensor],
    kinds: Tensor | None = None,
    dim: Optional[Tuple[int, ...]] = None,
    ignore_unused_parameters: bool = False,
    **kwargs,
) -> Callable[..., Tensor]:
    parameters, combinators = {}, {}

    for name, parameter in kwargs.items():
        if isinstance(parameter, Callable):
            combinators[name] = parameter
        elif isinstance(parameter, tuple) and isinstance(parameter[0], Callable):
            assert len(parameter) == 2

            combinators[name], parameters[name] = parameter[0], parameter[1]
        else:
            parameters[name] = parameter

    merged_dictionaries = functools.partial(
        _merge_dictionaries,
        ignore_unused_parameters=ignore_unused_parameters,
    )

    def mapped_fn(
        positions: Tensor,
        neighbor_list: _NeighborList,
        **dynamic_kwargs,
    ) -> Tensor:
        distance_fn = functools.partial(displacement_fn, **dynamic_kwargs)

        _kinds = dynamic_kwargs.get("kinds", kinds)

        normalization = 2.0

        if _is_neighbor_list_sparse(neighbor_list.format):
            distances = _map_bond(distance_fn)(
                positions[neighbor_list.indexes[0]],
                positions[neighbor_list.indexes[1]],
            )

            mask = torch.less(neighbor_list.indexes[0], positions.shape[0])

            if neighbor_list.format is _NeighborListFormat.ORDERED_SPARSE:
                normalization = 1.0
        else:
            distances = _map_neighbor(distance_fn)(
                positions,
                positions[neighbor_list.indexes],
            )

            mask = torch.less(neighbor_list.indexes, positions.shape[0])

        out = fn(
            distances,
            **_kwargs_to_neighbor_list_parameters(
                neighbor_list.format,
                neighbor_list.indexes,
                _kinds,
                merged_dictionaries(
                    parameters,
                    dynamic_kwargs,
                ),
                combinators,
            ),
        )

        if out.ndim > mask.ndim:
            mask = torch.reshape(
                mask,
                [*mask.shape, *([1] * (out.ndim - mask.ndim))],
            )

        out = torch.multiply(out, mask)

        if dim is None:
            return torch.divide(_safe_sum(out), normalization)

        if 0 in dim and 1 not in dim:
            raise ValueError

        if not _is_neighbor_list_sparse(neighbor_list.format):
            return torch.divide(_safe_sum(out, dim=dim), normalization)

        if 0 in dim:
            return _safe_sum(out, dim=[0, *[a - 1 for a in dim if a > 1]])

        if neighbor_list.format is _NeighborListFormat.ORDERED_SPARSE:
            raise ValueError

        return torch.divide(
            _segment_sum(
                _safe_sum(out, dim=[a - 1 for a in dim if a > 1]),
                neighbor_list.indexes[0],
                positions.shape[0],
            ),
            normalization,
        )

    return mapped_fn
