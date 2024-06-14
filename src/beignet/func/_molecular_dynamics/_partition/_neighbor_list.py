import dataclasses
import functools
from typing import Any, Callable, Optional

import numpy
import torch
from torch import Tensor

from .__cell_size import _cell_size
from .__clamp_indices import clamp_indices
from .__is_neighbor_list_format_valid import _is_neighbor_list_format_valid
from .__is_neighbor_list_sparse import _is_neighbor_list_sparse
from .__is_space_valid import _is_space_valid
from .__map_bond import _map_bond
from .__map_neighbor import _map_neighbor
from .__neighbor_list import _NeighborList
from .__neighbor_list_format import _NeighborListFormat
from .__neighbor_list_function_list import _NeighborListFunctionList
from .__neighboring_cell_lists import _neighboring_cell_lists
from .__normalize_cell_size import _normalize_cell_size
from .__partition_error import _PartitionError
from .__partition_error_kind import _PartitionErrorKind
from .__shift import _shift
from .__to_square_metric_fn import _to_square_metric_fn
from ._cell_list import cell_list


def neighbor_list(
    displacement_fn: Callable[[Tensor, Tensor], Tensor],
    space: Tensor,
    neighborhood_radius: float,
    maximum_distance: float = 0.0,
    buffer_size_multiplier: float = 1.25,
    disable_unit_list: bool = False,
    mask_self: bool = True,
    mask_fn: Optional[Callable[[Tensor], Tensor]] = None,
    normalized: bool = False,
    neighbor_list_format: _NeighborListFormat = _NeighborListFormat.DENSE,
    **_,
) -> _NeighborListFunctionList:
    _is_neighbor_list_format_valid(neighbor_list_format)

    space = space.detach()

    cutoff = neighborhood_radius + maximum_distance

    squared_cutoff = cutoff**2

    squared_maximum_distance = (maximum_distance / 2.0) ** 2

    metric_sq = _to_square_metric_fn(displacement_fn)

    def _neighbor_candidate_fn(shape: tuple[int, ...]) -> Tensor:
        return torch.broadcast_to(torch.arange(shape[0])[None, :], (shape[0], shape[0]))

    def _cell_list_neighbor_candidate_fn(unit_indexes_buffer, shape) -> Tensor:
        n, spatial_dimension = shape

        indexes = unit_indexes_buffer

        unit_indexes = [indexes]

        for dindex in _neighboring_cell_lists(spatial_dimension):
            if numpy.all(dindex == 0):
                continue

            unit_indexes += [_shift(indexes, dindex)]

        unit_indexes = torch.concatenate(unit_indexes, dim=-2)

        unit_indexes = unit_indexes[..., None, :, :]

        unit_indexes = torch.broadcast_to(
            unit_indexes, indexes.shape[:-1] + unit_indexes.shape[-2:]
        )

        def copy_values_from_cell(value, cell_value, cell_id):
            scatter_indices = torch.reshape(cell_id, (-1,))

            cell_value = torch.reshape(cell_value, (-1,) + cell_value.shape[-2:])

            value[scatter_indices] = cell_value

            return value

        neighbor_indexes = torch.zeros(
            (n + 1,) + unit_indexes.shape[-2:], dtype=torch.int32
        )

        neighbor_indexes = copy_values_from_cell(
            neighbor_indexes, unit_indexes, indexes
        )

        return neighbor_indexes[:-1, :, 0]

    def mask_self_fn(idx: Tensor) -> Tensor:
        return torch.where(
            idx
            == torch.reshape(
                torch.arange(idx.shape[0], dtype=torch.int32),
                (idx.shape[0], 1),
            ),
            idx.shape[0],
            idx,
        )

    def prune_dense_neighbor_list(
        positions: Tensor, indexes: Tensor, **kwargs
    ) -> Tensor:
        displacement_fn = functools.partial(metric_sq, **kwargs)

        displacement_fn = _map_neighbor(displacement_fn)

        try:
            neighbor_positions = positions[indexes]

        except:
            neighbor_positions = torch.zeros(indexes.shape[0], indexes.shape[1], positions.shape[1])

        displacements = displacement_fn(positions, neighbor_positions)

        mask = (displacements < squared_cutoff) & (indexes < positions.shape[0])

        output_indexes = positions.shape[0] * torch.ones(
            indexes.shape, dtype=torch.int32
        )

        cumsum = torch.cumsum(mask, dim=1)

        index = torch.where(mask, cumsum - 1, indexes.shape[1] - 1)

        p_index = torch.arange(indexes.shape[0])[:, None]

        output_indexes[p_index, index] = indexes

        maximum_occupancy = torch.max(cumsum[:, -1])

        return output_indexes, maximum_occupancy

    def prune_sparse_neighbor_list(
        position: Tensor, idx: Tensor, **kwargs
    ) -> tuple[Tensor, Any]:
        displacement_fn = functools.partial(metric_sq, **kwargs)

        displacement_fn = _map_bond(displacement_fn)

        sender_idx = torch.broadcast_to(
            torch.arange(position.shape[0])[:, None], idx.shape
        )

        sender_idx = torch.reshape(sender_idx, (-1,))

        receiver_idx = torch.reshape(idx, (-1,))

        mask_before_indexing = (receiver_idx < position.shape[0]) & (
            sender_idx < position.shape[0]
        )
        sender_idx = sender_idx[mask_before_indexing]
        receiver_idx = receiver_idx[mask_before_indexing]

        distances = displacement_fn(position[sender_idx], position[receiver_idx])

        mask = (
            (distances < squared_cutoff)
            & (receiver_idx < position.shape[0])
            & (sender_idx < position.shape[0])
        )

        if neighbor_list_format is _NeighborListFormat.ORDERED_SPARSE:
            mask = mask & (receiver_idx < sender_idx)

        out_idx = position.shape[0] * torch.ones(receiver_idx.shape, dtype=torch.int32)

        cumsum = torch.cumsum(mask.reshape(-1), dim=0)

        index = torch.where(mask, cumsum - 1, len(receiver_idx) - 1)

        out_idx[index] = receiver_idx.int()

        receiver_idx = out_idx[index]

        out_idx[index] = sender_idx.int()

        sender_idx = out_idx[index]

        max_occupancy = cumsum[-1]

        return torch.stack([receiver_idx, sender_idx]), max_occupancy

    def _neighbors_fn(
        positions: Tensor, neighbors=None, extra_capacity: int = 0, **kwargs
    ) -> _NeighborList:
        # print(f"extra_capacity: {extra_capacity}")
        def _fn(position_and_error, maximum_size=None):
            reference_positions, err = position_and_error

            n = reference_positions.shape[0]
            # print(f"n: {n}")

            buffer_fn = None

            unit_list = None

            item_size = None

            # print(f"disable_unit_list: {disable_unit_list}")

            if not disable_unit_list:
                if neighbors is None:
                    _space = kwargs.get("space", space)

                    item_size = cutoff
                    # print(f"normalized: {normalized}")
                    if normalized:
                        if not torch.all(positions < 1):
                            raise ValueError(
                                "Positions are not normalized. Ensure torch.all(positions < 1)."
                            )

                        err = err.update(
                            _PartitionErrorKind.MALFORMED_BOX,
                            _is_space_valid(_space),
                        )

                        item_size = _normalize_cell_size(_space, cutoff)

                        _space = 1.0

                    if torch.all(item_size < _space / 3.0):
                        # print("here")
                        buffer_fn = cell_list(_space, item_size, buffer_size_multiplier)

                        unit_list = buffer_fn.setup_fn(
                            reference_positions, excess_buffer_size=extra_capacity
                        )
                        # print(f"positions: {unit_list.positions_buffer.shape[0]}")
                        # print(f"indexes: {unit_list.indexes.shape[0]}")
                else:
                    item_size = neighbors.item_size

                    buffer_fn = neighbors.buffer_fn

                    if buffer_fn is not None:
                        unit_list = buffer_fn.update_fn(
                            reference_positions, neighbors.units_buffer_size
                        )

            if unit_list is None:
                units_buffer_size = None

                indexes = _neighbor_candidate_fn(reference_positions.shape)
            else:
                err = err.update(
                    _PartitionErrorKind.CELL_LIST_OVERFLOW,
                    unit_list.exceeded_maximum_size,
                )

                indexes = _cell_list_neighbor_candidate_fn(
                    unit_list.indexes, reference_positions.shape
                )

                units_buffer_size = unit_list.size

                print(f"idx: {indexes.shape[0]}")
                print(f"cl capacity: {units_buffer_size}")

            if mask_self:
                indexes = mask_self_fn(indexes)
            if mask_fn is not None:
                indexes = mask_fn(indexes)

            if _is_neighbor_list_sparse(neighbor_list_format):
                indexes, occupancy = prune_sparse_neighbor_list(
                    reference_positions, indexes, **kwargs
                )
            else:
                # print(f"position: {reference_positions}")

                indexes, occupancy = prune_dense_neighbor_list(
                    reference_positions, indexes, **kwargs
                )

            if maximum_size is None:
                if not _is_neighbor_list_sparse(neighbor_list_format):
                    _extra_capacity = extra_capacity
                else:
                    _extra_capacity = n * extra_capacity

                maximum_size = int(occupancy * buffer_size_multiplier + _extra_capacity)

                if maximum_size > indexes.shape[-1]:
                    maximum_size = indexes.shape[-1]

                if not _is_neighbor_list_sparse(neighbor_list_format):
                    capacity_limit = n - 1 if mask_self else n
                elif neighbor_list_format is _NeighborListFormat.SPARSE:
                    capacity_limit = n * (n - 1) if mask_self else n**2
                else:
                    capacity_limit = n * (n - 1) // 2

                if maximum_size > capacity_limit:
                    maximum_size = capacity_limit

            indexes = indexes[:, :maximum_size]

            if neighbors is None:
                update_fn = _neighbors_fn
            else:
                update_fn = neighbors.update_fn

            partition_error = err.update(
                _PartitionErrorKind.NEIGHBOR_LIST_OVERFLOW,
                occupancy > maximum_size,
            )

            return _NeighborList(
                buffer_fn=buffer_fn,
                indexes=indexes,
                item_size=item_size,
                maximum_size=maximum_size,
                format=neighbor_list_format,
                partition_error=partition_error,
                reference_positions=reference_positions,
                units_buffer_size=units_buffer_size,
                update_fn=update_fn,
            )

        updated_neighbors = neighbors

        # print(f"nbrs: {updated_neighbors}")

        if updated_neighbors is None:
            return _fn(
                (
                    positions,
                    _PartitionError(torch.zeros([], dtype=torch.uint8)),
                )
            )

        neighbor_fn = functools.partial(
            _fn,
            maximum_size=updated_neighbors.maximum_size,
        )

        if space and not disable_unit_list:
            if not normalized:
                raise ValueError

            current_unit_size = _cell_size(
                torch.tensor(1.0),
                updated_neighbors.item_size,
            )

            updated_unit_size = _cell_size(
                torch.tensor(1.0),
                _normalize_cell_size(space, cutoff),
            )

            error = updated_neighbors.partition_error.update(
                _PartitionErrorKind.CELL_SIZE_TOO_SMALL,
                updated_unit_size > current_unit_size,
            )

            error = error.update(
                _PartitionErrorKind.MALFORMED_BOX,
                _is_space_valid(space),
            )

            updated_neighbors = dataclasses.replace(
                updated_neighbors, partition_error=error
            )

        displacement_fn = functools.partial(metric_sq, **kwargs)

        displacement_fn = _map_bond(displacement_fn)

        predicate = torch.any(
            displacement_fn(positions, updated_neighbors.reference_positions)
            > squared_maximum_distance
        )

        if predicate:
            return (positions, updated_neighbors.partition_error)

        else:
            return neighbor_fn((positions, updated_neighbors.partition_error))

    def setup_fn(positions: Tensor, extra_capacity: int = 0, **kwargs):
        return _neighbors_fn(positions, extra_capacity=extra_capacity, **kwargs)

    def update_fn(positions: Tensor, neighbors, **kwargs):
        return _neighbors_fn(positions, neighbors, **kwargs)

    return _NeighborListFunctionList(setup_fn=setup_fn, update_fn=update_fn)
