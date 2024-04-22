from typing import Callable

import torch
from torch import Tensor

from .__cell_dimensions import _cell_dimensions
from .__cell_list import _CellList
from .__cell_list_function_list import _CellListFunctionList
from .__hash_constants import _hash_constants
from .__iota import _iota
from .__particles_per_cell import _particles_per_cell
from .__segment_sum import _segment_sum
from .__unflatten_cell_buffer import _unflatten_cell_buffer


def cell_list(
    size: Tensor,
    minimum_unit_size: float,
    buffer_size_multiplier: float = 1.25,
) -> _CellListFunctionList:
    if not isinstance(size, Tensor):
        size = torch.tensor(size, dtype=torch.float32)

    if size.ndim == 1:
        size = torch.reshape(size, [1, -1])

    def fn(
        positions: Tensor,
        excess: tuple[bool, int, Callable[..., _CellList]] | None = None,
        excess_buffer_size: int = 0,
        **kwargs,
    ) -> _CellList:
        spatial_dimension = positions.shape[1]

        if spatial_dimension not in {2, 3}:
            raise ValueError

        _, unit_size, units_per_side, unit_count = _cell_dimensions(
            spatial_dimension,
            size,
            minimum_unit_size,
        )

        if excess is None:
            buffer_size = int(
                torch.max(_particles_per_cell(positions, size, unit_size))
                * buffer_size_multiplier
            )

            buffer_size = buffer_size + excess_buffer_size

            exceeded_maximum_size = False

            update_fn = fn
        else:
            buffer_size, exceeded_maximum_size, update_fn = excess

        positions_buffer = torch.zeros(
            [unit_count * buffer_size, spatial_dimension],
            device=positions.device,
            dtype=positions.dtype,
        )

        indexes = positions.shape[0] * torch.ones(
            [unit_count * buffer_size, 1],
            device=positions.device,
            dtype=torch.int32,
        )

        parameters = {}

        for name, parameter in kwargs.items():
            if not isinstance(parameter, Tensor):
                raise ValueError(
                    (
                        f'Data must be specified as an ndarray. Found "{name}" '
                        f"with type {type(parameter)}."
                    )
                )

            if parameter.shape[0] != positions.shape[0]:
                raise ValueError

            if parameter.ndim > 1:
                kwarg_shape = parameter.shape[1:]
            else:
                kwarg_shape = (1,)

            parameters[name] = 100000 * torch.ones(
                (unit_count * buffer_size,) + kwarg_shape, parameter.dtype
            )

        hashes = torch.sum(
            (positions / unit_size).to(dtype=torch.int32)
            * _hash_constants(spatial_dimension, units_per_side).to(
                device=positions.device
            ),
            dim=1,
        )

        sort_map = torch.argsort(hashes)

        sorted_parameters = {}

        for name, parameter in kwargs.items():
            sorted_parameters[name] = parameter[sort_map]

        sorted_unit_indexes = hashes[sort_map] * buffer_size + torch.remainder(
            _iota(
                ((positions.shape[0]),),
                device=positions.device,
                dtype=torch.int32,
            ),
            buffer_size,
        )

        positions_buffer[sorted_unit_indexes] = positions[sort_map]

        indexes[sorted_unit_indexes] = torch.reshape(
            _iota(((positions.shape[0]),), dtype=torch.int32).to(
                device=positions.device
            )[sort_map],
            [(positions.shape[0]), 1],
        )

        positions_buffer = _unflatten_cell_buffer(
            positions_buffer, units_per_side, spatial_dimension
        )

        indexes = _unflatten_cell_buffer(indexes, units_per_side, spatial_dimension)

        for name, parameter in sorted_parameters.items():
            if parameter.ndim == 1:
                parameter = torch.reshape(parameter, parameter.shape + (1,))

            parameters[name][sorted_unit_indexes] = parameter

            parameters[name] = _unflatten_cell_buffer(
                parameters[name], units_per_side, spatial_dimension
            )

        exceeded_maximum_size = exceeded_maximum_size | (
            torch.max(_segment_sum(torch.ones_like(hashes), hashes, unit_count))
            > buffer_size
        )

        return _CellList(
            exceeded_maximum_size=exceeded_maximum_size,
            indexes=indexes,
            parameters=parameters,
            positions_buffer=positions_buffer,
            size=buffer_size,
            item_size=unit_size,
            update_fn=update_fn,
        )

    def setup_fn(
        positions: Tensor,
        excess_buffer_size: int = 0,
        **kwargs,
    ) -> _CellList:
        return fn(positions, excess_buffer_size=excess_buffer_size, **kwargs)

    def update_fn(
        positions: Tensor,
        buffer: int | _CellList,
        **kwargs,
    ) -> _CellList:
        if isinstance(buffer, int):
            return fn(positions, (buffer, False, fn), **kwargs)

        return fn(
            positions,
            (buffer.size, buffer.exceeded_maximum_size, buffer.update_fn),
            **kwargs,
        )

    return _CellListFunctionList(setup_fn=setup_fn, update_fn=update_fn)
