import dataclasses
import functools
import math
import operator
from enum import Enum, IntEnum
from typing import Dict, Callable, Any, Optional, Generator

import torch
from torch import Tensor

from beignet.func.__dataclass import _dataclass
from beignet.func._static_field import static_field

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PartitionErrorCode(IntEnum):
    """An enum specifying different error codes.

    Attributes:
      NONE: Means that no error was encountered during simulation.
      NEIGHBOR_LIST_OVERFLOW: Indicates that the neighbor list was not large
        enough to contain all of the particles. This should indicate that it is
        necessary to allocate a new neighbor list.
      CELL_LIST_OVERFLOW: Indicates that the cell list was not large enough to
        contain all of the particles. This should indicate that it is necessary
        to allocate a new cell list.
      CELL_SIZE_TOO_SMALL: Indicates that the size of cells in a cell list was
        not large enough to properly capture particle interactions. This
        indicates that it is necessary to allcoate a new cell list with larger
        cells.
      MALFORMED_BOX: Indicates that a box matrix was not properly upper
        triangular.
    """

    NONE = 0
    NEIGHBOR_LIST_OVERFLOW = 1 << 0
    CELL_LIST_OVERFLOW = 1 << 1
    CELL_SIZE_TOO_SMALL = 1 << 2
    MALFORMED_BOX = 1 << 3


PEC = PartitionErrorCode


@_dataclass
class _PartitionError:
    r"""A class to represent and manage partition errors with specific error codes.

    Attributes:
    -----------
    code : Tensor
        A tensor representing the error code.

    Methods:
    --------
    update(bit: bytes, predicate: Tensor) -> "_PartitionError"
        Update the error code based on a predicate and a new bit.

    __str__() -> str
        Provide a human-readable string representation of the error.

    __repr__() -> str
        Alias for __str__().
    """

    code: Tensor

    def update(self, bit: bytes, predicate: Tensor) -> "_PartitionError":
        r"""Update the error code based on a predicate and a new bit.

        Parameters:
        -----------
        bit : bytes
            The bit to be combined with the existing error code.
        predicate : Tensor
            A tensor that determines where the bit should be applied.

        Returns:
        --------
        _PartitionError
            A new instance of `_PartitionError` with the updated error code.
        """
        zero = torch.zeros([], dtype=torch.uint8)

        bit = bit

        return _PartitionError(code=self.code | torch.where(predicate, bit, zero))

    def __str__(self) -> str:
        r"""Provide a human-readable string representation of the error.

        Returns:
        --------
        str
            A string describing the error.

        Raises:
        -------
        ValueError
            If the error code is unexpected or not recognized.
        """
        if not torch.any(self.code):
            return ""

        if torch.any(self.code & _PartitionErrorKind.NEIGHBOR_LIST_OVERFLOW):
            return "Partition Error: Neighbor list buffer overflow."

        if torch.any(self.code & _PartitionErrorKind.CELL_LIST_OVERFLOW):
            return "Partition Error: Cell list buffer overflow"

        if torch.any(self.code & _PartitionErrorKind.CELL_SIZE_TOO_SMALL):
            return "Partition Error: Cell size too small"

        if torch.any(self.code & _PartitionErrorKind.MALFORMED_BOX):
            return "Partition Error: Incorrect box format. Expecting upper triangular."

        raise ValueError(f"Unexpected Error Code {self.code}.")

    __repr__ = __str__


class _PartitionErrorKind(IntEnum):
    r"""An enumeration representing different kinds of partition errors in a particle simulation.

    Attributes
    ----------
    NONE : int
        No error.
    NEIGHBOR_LIST_OVERFLOW : int
        Indicates that the neighbor list has overflowed.
    CELL_LIST_OVERFLOW : int
        Indicates that the cell list has overflowed.
    CELL_SIZE_TOO_SMALL : int
        Indicates that the cell size is too small.
    MALFORMED_BOX : int
        Indicates that the simulation box is malformed.
    """

    NONE = 0
    NEIGHBOR_LIST_OVERFLOW = 1 << 0
    CELL_LIST_OVERFLOW = 1 << 1
    CELL_SIZE_TOO_SMALL = 1 << 2
    MALFORMED_BOX = 1 << 3


@_dataclass
class _CellList:
    r"""Stores the spatial partition of a system into a cell list.

    See :meth:`cell_list` for details on the construction / specification.
    Cell list buffers all have a common shape, S, where
    * `S = [cell_count_x, cell_count_y, cell_capacity]`
    * `S = [cell_count_x, cell_count_y, cell_count_z, cell_capacity]`
    in two- and three-dimensions respectively. It is assumed that each cell has
    the same capacity.

    Attributes:
    positions_buffer: An ndarray of floating point positions with shape
      `S + [spatial_dimension]`.
    indexes: An ndarray of int32 particle ids of shape `S`. Note that empty
      slots are specified by `id = N` where `N` is the number of particles in
      the system.
    parameters: A dictionary of ndarrays of shape `S + [...]`. This contains
      side data placed into the cell list.
    exceeded_maximum_size: A boolean specifying whether or not the cell list
      exceeded the maximum allocated capacity.
    size: An integer specifying the maximum capacity of each cell in
      the cell list.
    item size: A tensor specifying the size of each cell in the cell list.
    update_fn: A function that updates the cell list at a fixed capacity.
    """

    exceeded_maximum_size: Tensor
    indexes: Tensor
    item_size: float = static_field()
    parameters: Dict[str, Tensor]
    positions_buffer: Tensor
    size: int = static_field()
    update_fn: Callable[..., "_CellList"] = static_field()

    def update(self, positions: Tensor, **kwargs) -> "_CellList":
        return self.update_fn(
            positions,
            [
                self.size,
                self.exceeded_maximum_size,
                self.update_fn,
            ],
            **kwargs,
        )


@_dataclass
class _CellListFunctionList:
    r"""A dataclass that encapsulates functions for setting up and updating a cell list.

    Attributes
    ----------
    setup_fn : Callable[..., _CellList]
        A function that sets up and returns a `_CellList` object.
    update_fn : Callable[[Tensor, Union[_CellList, int]], _CellList]
        A function that updates a `_CellList` object given a tensor and either an existing `_CellList` or an integer.

    Methods
    -------
    __iter__()
        Returns an iterator over the setup and update functions.
    """

    setup_fn: Callable[..., _CellList] = static_field()

    update_fn: Callable[[Tensor, _CellList | int], _CellList] = static_field()

    def __iter__(self):
        return iter([self.setup_fn, self.update_fn])


class _NeighborListFormat(Enum):
    r"""An enumeration representing the format of a neighbor list.

    Attributes
    ----------
    DENSE : int
        Represents a dense neighbor list format.
    ORDERED_SPARSE : int
        Represents an ordered sparse neighbor list format.
    SPARSE : int
        Represents a sparse neighbor list format.
    """

    DENSE = 0
    ORDERED_SPARSE = 1
    SPARSE = 2


@_dataclass
class _NeighborList:
    r"""A dataclass representing a neighbor list used in particle simulations.

    Attributes
    ----------
    buffer_fn : Callable[[Tensor, _CellList], _CellList]
        A function to buffer the cell list.
    indexes : Tensor
        A tensor containing the indexes of neighbors.
    item_size : float or None
        The size of each item in the neighbor list.
    maximum_size : int
        The maximum size of the neighbor list.
    format : _NeighborListFormat
        The format of the neighbor list.
    partition_error : _PartitionError
        An object representing partition errors.
    reference_positions : Tensor
        A tensor containing the reference positions of particles.
    units_buffer_size : int or None
        The buffer size in units.
    update_fn : Callable[[Tensor, "_NeighborList", Any], "_NeighborList"]
        A function to update the neighbor list.

    Methods
    -------
    update(positions: Tensor, **kwargs) -> "_NeighborList"
        Updates the neighbor list with new positions.
    did_buffer_overflow() -> bool
        Checks if the buffer overflowed.
    cell_size_too_small() -> bool
        Checks if the cell size is too small.
    malformed_box() -> bool
        Checks if the box is malformed.
    """

    buffer_fn: Callable[[Tensor, _CellList], _CellList] = static_field()
    indexes: Tensor
    item_size: float | None = static_field()
    maximum_size: int = static_field()
    format: _NeighborListFormat = static_field()
    partition_error: _PartitionError
    reference_positions: Tensor
    units_buffer_size: int | None = static_field()
    update_fn: Callable[[Tensor, "_NeighborList", Any], "_NeighborList"] = (
        static_field()
    )

    def update(self, positions: Tensor, **kwargs) -> "_NeighborList":
        return self.update_fn(positions, self, **kwargs)

    @property
    def did_buffer_overflow(self) -> bool:
        return (
            self.partition_error.code
            & (PEC.NEIGHBOR_LIST_OVERFLOW | PEC.CELL_LIST_OVERFLOW)
        ).item() != 0

    @property
    def cell_size_too_small(self) -> bool:
        return (self.partition_error.code & PEC.CELL_SIZE_TOO_SMALL).item() != 0

    @property
    def malformed_box(self) -> bool:
        return (self.partition_error.code & PEC.MALFORMED_BOX).item() != 0


@_dataclass
class _NeighborListFunctionList:
    r"""
    A dataclass that encapsulates functions for setting up and updating a neighbor list.

    Attributes
    ----------
    setup_fn : Callable[..., _NeighborList]
        A function that sets up and returns a `_NeighborList` object.
    update_fn : Callable[[Tensor, _NeighborList], _NeighborList]
        A function that updates a `_NeighborList` object given a tensor and an existing `_NeighborList`.

    Methods
    -------
    __iter__()
        Returns an iterator over the setup and update functions.
    """

    setup_fn: Callable[..., _NeighborList] = static_field()

    update_fn: Callable[[Tensor, _NeighborList], _NeighborList] = static_field()

    def __iter__(self):
        return iter((self.setup_fn, self.update_fn))


def distance(dR: Tensor) -> Tensor:
    """Computes distances.

    Args:
      dR: Matrix of displacements; `Tensor(shape=[..., spatial_dim])`.
    Returns:
      Matrix of distances; `Tensor(shape=[...])`.
    """
    dr = square_distance(dR)

    return safe_mask(dr > 0, torch.sqrt, dr)


def _iota(shape: tuple[int, ...], dim: int = 0, **kwargs) -> Tensor:
    r"""Generate a tensor with a specified shape where elements along the given dimension
    are sequential integers starting from 0.

    Parameters
    ----------
    shape : tuple[int, ...]
        The shape of the resulting tensor.
    dim : int, optional
        The dimension along which to vary the values (default is 0).

    Returns
    -------
    Tensor
        A tensor of the specified shape with sequential integers along the specified dimension.

    Raises
    ------
    IndexError
        If `dim` is out of the range of `shape`.
    """
    dimensions = []

    for index, _ in enumerate(shape):
        if index != dim:
            dimension = 1

        else:
            dimension = shape[index]

        dimensions = [*dimensions, dimension]

    return torch.arange(shape[dim], **kwargs).view(*dimensions).expand(*shape)


def _hash_constants(spatial_dimensions: int, cells_per_side: Tensor) -> Tensor:
    r"""Compute constants used for hashing in a spatial partitioning scheme.

    The function calculates constants that help in determining the hash value
    for a given cell in an N-dimensional grid, based on the number of cells
    per side in each dimension.

    Parameters
    ----------
    spatial_dimensions : int
        The number of spatial dimensions.
    cells_per_side : Tensor
        A 1D tensor indicating the number of cells per side in each dimension.
        If the size of the tensor is 0, it is assumed that the grid is uniform
        in all dimensions, and this single value will be used for all dimensions.
        If the number of elements matches `spatial_dimensions`, these values
        specify the number of cells per side for each corresponding dimension.

    Returns
    -------
    Tensor
        A 2D tensor of shape (1, spatial_dimensions) containing the computed hash
        constants, used to index cells in the grid.

    Raises
    ------
    ValueError
        If the size of `cells_per_side` is not zero or `spatial_dimensions`.
    """
    if cells_per_side.numel() == 1:
        constants = [[cells_per_side**dim for dim in range(spatial_dimensions)]]
        return torch.tensor(constants, device=device, dtype=torch.int32)

    elif cells_per_side.numel() == spatial_dimensions:
        one = torch.tensor([[1]], device=device, dtype=torch.int32)
        cells_per_side = torch.cat((one, cells_per_side[:, :-1]), dim=1)
        return torch.cumprod(cells_per_side, dim=1).squeeze()

    else:
        raise ValueError(
            "Cells per side must either: have 0 dimensions, or be the same size as spatial dimensions."
        )


def _map_bond(metric_or_displacement: Callable) -> Callable:
    r"""Map a distance function over batched start and end positions.

    Parameters:
    -----------
    distance_fn : callable
        A function that computes the distance between two positions.

    Returns:
    --------
    wrapper : callable
        A wrapper function that applies `distance_fn` to each pair of start and end positions
        in the batch.
    """
    return torch.vmap(metric_or_displacement, (0, 0), 0)


def _map_neighbor(metric_or_displacement: Callable) -> Callable:
    r"""Vectorizes a metric or displacement function over neighborhoods."""

    def wrapped_fn(Ra, Rb, **kwargs):
        return torch.vmap(torch.vmap(metric_or_displacement, (0, None)))(
            Rb, Ra, **kwargs
        )

    return wrapped_fn


def map_product(metric_or_displacement: Callable) -> Callable:
    r"""Vectorizes a metric or displacement function over all pairs."""
    outer_vmap = torch.vmap(torch.vmap(metric_or_displacement, (0, None)), (None, 0))

    return outer_vmap


def metric(distance_fn: Callable) -> Callable:
    r"""Takes a displacement function and creates a metric..

    Parameters:
    -----------
    distance_fn : callable
        A function that computes the distance between two positions.

    Returns:
    --------
    wrapper : callable
        A wrapper function that applies `distance_fn` to each pair of start and end positions
        in the batch.
    """
    return lambda Ra, Rb, **kwargs: distance(distance_fn(Ra, Rb, **kwargs))


def safe_index(
    array: Tensor, indices: Tensor, indices_b: Optional[Tensor] = None
) -> Tensor:
    r"""Safely index into a tensor, clamping out-of-bounds indices to the nearest valid index.

    Parameters
    ----------
    array : Tensor
        The tensor to index.
    indices : Tensor
        The indices to use for indexing.
    indices_b : Tensor
        Another Tensor of indices to for advanced indexing.

    Returns
    -------
    Tensor
        The resulting tensor after indexing.
    """
    if indices_b is not None:
        indices = torch.clamp(indices, 0, array.size(0) - 1)

        indices_b = torch.clamp(indices_b, 0, array.size(1) - 1)

        return array[indices.to(torch.long), indices_b.to(torch.long)]

    else:
        indices = torch.clamp(indices, 0, array.size(0) - 1)

        return array[indices]


def safe_mask(
    mask: Tensor, fn: Callable, operand: Tensor, placeholder: Any = 0
) -> Tensor:
    r"""Applies a function to elements of a tensor where a mask is True, and replaces elements where the mask is False with a placeholder.

    Parameters
    ----------
    mask : Tensor
        A boolean tensor indicating which elements to apply the function to.
    fn : Callable[[Tensor], Tensor]
        The function to apply to the masked elements.
    operand : Tensor
        The tensor to apply the function to.
    placeholder : Any, optional
        The value to use for elements where the mask is False (default is 0).

    Returns
    -------
    Tensor
        A tensor with the function applied to the masked elements and the placeholder value elsewhere.
    """
    masked = torch.where(mask, operand, 0)

    return torch.where(mask, fn(masked), placeholder)


def _segment_sum(
    input: Tensor,
    indexes: Tensor,
    n: Optional[int] = None,
    **kwargs,
) -> Tensor:
    r"""Computes the sum of segments of a tensor along the first dimension.

    Parameters
    ----------
    input : Tensor
        A tensor containing the input values to be summed.
    indexes : Tensor
        A 1D tensor containing the segment indexes for summation.
        Should have the same length as the first dimension of the `input` tensor.
    n : Optional[int], optional
        The number of segments, by default `n` is set to `max(indexes) + 1`.

    Returns
    -------
    Tensor
        A tensor where each entry contains the sum of the corresponding segment
        from the `input` tensor.
    """
    if input.size(0) != indexes.size(0):
        raise ValueError(
            "The length of the indexes tensor must match the size of the first dimension of the input tensor."
        )

    if n is None:
        n = indexes.max().item() + 1

    output = torch.zeros(n, *input.shape[1:], device=input.device)

    indexes = torch.clamp(indexes, 0, output.size(0) - 1)

    if indexes.dim() != input.dim():
        indexes = indexes.unsqueeze(1).expand(-1, input.size(1))

    return output.scatter_add(0, indexes.to(torch.int64), input.to(torch.float32)).to(
        **kwargs
    )


def _shift(a: Tensor, b: Tensor) -> Tensor:
    r"""Shifts a tensor `a` along dimensions specified in `b`.

    The shift can be applied in up to three dimensions (x, y, z).
    Positive values in `b` indicate a forward shift, while negative values indicate
    a backward shift.

    Parameters
    ----------
    a : Tensor
      The input tensor to be shifted.
    b : Tensor
      A tensor of two or three elements specifying the shift amount for each dimension.

    Returns
    -------
    Tensor
      The shifted tensor.
    """
    return torch.roll(a, shifts=tuple(b), dims=tuple(range(len(b))))


def square_distance(input: Tensor) -> Tensor:
    """Computes square distances.

    Args:
    input: Matrix of displacements; `Tensor(shape=[..., spatial_dim])`.
    Returns:
    Matrix of squared distances; `Tensor(shape=[...])`.
    """
    return torch.sum(input**2, dim=-1)


def _to_square_metric_fn(
    fn: Callable[[Tensor, Tensor, Any], Tensor],
) -> Callable[[Tensor, Tensor, Any], Tensor]:
    r"""Converts a given distance function to a squared distance metric.

    The function tries to apply the given distance function `fn` to positions in
    one to three dimensions to determine if the output is scalar or vector.
    Based on this, it returns a new function that computes the squared distance.

    Parameters
    ----------
    fn : Callable[[Tensor, Tensor, Any], Tensor]
        A function that computes distances between two tensors.

    Returns
    -------
    Callable[[Tensor, Tensor, Any], Tensor]
        A function that computes the squared distance metric using the given distance function.
    """
    for dimension in range(1, 4):
        try:
            positions = torch.rand([dimension], dtype=torch.float32)

            distances = fn(positions, positions, t=0)  # type: ignore[no-untyped-def]

            if distances.ndim == 0:

                def square_metric(a: Tensor, b: Tensor, **kwargs) -> Tensor:
                    return torch.square(fn(a, b, **kwargs))
            else:

                def square_metric(a: Tensor, b: Tensor, **kwargs) -> Tensor:
                    return torch.sum(torch.square(fn(a, b, **kwargs)), dim=-1)

            return square_metric

        except TypeError:
            continue

        except ValueError:
            continue

    raise ValueError


def _unflatten_cell_buffer(
    buffer: Tensor, cells_per_side: [int, float, Tensor], dim: int
):
    r"""Reshape a flat buffer into a multidimensional cell buffer.

    Parameters
    ----------
    buffer : Tensor
        The input flat buffer tensor to be reshaped.
    cells_per_side : Tensor or int or float
        The number of cells per side in each dimension.
    dim : int
        The number of spatial dimensions.

    Returns
    -------
    Tensor
        The reshaped buffer tensor.
    """
    # if cells per side is an int, float, or a tensor with 0 dimensions (scalar) -> return tuple
    if (
        isinstance(cells_per_side, int)
        or isinstance(cells_per_side, float)
        or (torch.is_tensor(cells_per_side) and not cells_per_side.shape)
    ):
        cells_per_side = (int(cells_per_side),) * dim

    # if cells per side is a 1D tensor, reverse the tensor, and place each elem into a tuple
    elif torch.is_tensor(cells_per_side) and len(cells_per_side.shape) == 1:
        reversed_cells_per_side = torch.flip(cells_per_side, [0])

        cells_per_side = tuple(int(x) for x in reversed_cells_per_side)

    elif torch.is_tensor(cells_per_side) and len(cells_per_side.shape) == 2:
        reversed_first_row = torch.flip(cells_per_side[0], [0])

        cells_per_side = tuple(int(x) for x in reversed_first_row)

    else:
        raise ValueError()

    new_shape = cells_per_side + (-1,) + buffer.shape[1:]

    # Reshape the buffer with the new shape
    reshaped_buffer = buffer.reshape(new_shape)

    return reshaped_buffer


def _cell_dimensions(
    spatial_dimension: int,
    box_size: Tensor,
    minimum_cell_size: float,
) -> (Tensor, Tensor, Tensor, int):
    r"""Compute the number of cells-per-side and total number of cells in a box.

    Parameters:
    -----------
    spatial_dimension : int
        The spatial dimension of the box (e.g., 2 for 2D, 3 for 3D).
    box_size : Tensor or scalar
        The size of the box. Can be a scalar or a Tensor with dimensions 0, 1, or 2.
    minimum_cell_size : float
        The minimum size of the cells.

    Returns:
    --------
    box_size : Tensor
        The (possibly modified) size of the box.
    cell_size : Tensor
        The size of the cells.
    cells_per_side : Tensor
        The number of cells per side.
    cell_count : int
        The total number of cells in the box.

    Raises:
    -------
    ValueError
        If the box size is less than the minimum cell size.
        If the box is not at least 3x the size of the grid spacing in each dimension.
        If the box is not a scalar, a vector, or a matrix.
    """
    if isinstance(box_size, (int, float)):
        box_size = float(box_size)

        if box_size < minimum_cell_size:
            raise ValueError("Box size must be at least as large as minimum cell size.")

    if isinstance(box_size, Tensor):
        if box_size.dtype in {torch.int32, torch.int64}:
            box_size = box_size.float()

    if isinstance(box_size, Tensor):
        cells_per_side = torch.floor(box_size / minimum_cell_size)

        cell_size = box_size / cells_per_side

        cells_per_side = cells_per_side.to(torch.int32)

        if box_size.dim() == 1 or box_size.dim() == 2:
            assert box_size.numel() == spatial_dimension

            flattened_cells_per_side = cells_per_side.view(-1)

            for cells in flattened_cells_per_side:
                if cells.item() < 3:
                    raise ValueError(
                        "Box must be at least 3x the size of the grid spacing in each dimension."
                    )

            cell_count = functools.reduce(
                operator.mul,
                flattened_cells_per_side,
                1,
            )

        elif box_size.dim() == 0:
            cell_count = cells_per_side**spatial_dimension

        else:
            raise ValueError(
                f"Box must be either: a scalar, a vector, or a matrix. Found {box_size}."
            )

    else:
        cells_per_side = math.floor(box_size / minimum_cell_size)

        cell_size = box_size / cells_per_side

        cell_count = cells_per_side**spatial_dimension

    return box_size, cell_size, cells_per_side, int(cell_count)


def _cell_size(box: Tensor, minimum_unit_size: Tensor) -> Tensor:
    r"""Compute the size of cells within a box based on the minimum unit size.

    Parameters:
    -----------
    box : Tensor
        The size of the box. This must be a Tensor.
    minimum_unit_size : Tensor
        The minimum size of the units (cells). This must be a Tensor of the same shape as `box` or a scalar Tensor.

    Returns:
    --------
    Tensor
        The size of the cells in the box.

    Raises:
    -------
    ValueError
        If the box and minimum unit size do not have the same shape and `minimum_unit_size` is not a scalar.
    """
    if box.shape == minimum_unit_size.shape or minimum_unit_size.ndim == 0:
        return box / torch.floor(box / minimum_unit_size)

    else:
        raise ValueError("Box and minimum unit size must be of the same shape.")


def _estimate_cell_capacity(
    positions: Tensor, size: Tensor, unit_size: float, buffer_size_multiplier: float
) -> int:
    r"""Estimates the capacity of a cell based on particle positions, cell size, unit size, and a buffer size multiplier.

    Parameters
    ----------
    positions : Tensor
        A tensor containing the positions of particles.
    size : Tensor
        A tensor representing the size of the cell.
    unit_size : float
        The size of a single unit within the cell.
    buffer_size_multiplier : float
        A multiplier to account for buffer space in the cell capacity.

    Returns
    -------
    int
        The estimated capacity of the cell, adjusted by the buffer size multiplier.
    """
    cell_capacity = torch.max(_particles_per_cell(positions, size, unit_size))

    return int(cell_capacity * buffer_size_multiplier)


def _is_neighbor_list_format_valid(neighbor_list_format: _NeighborListFormat):
    r"""Check if the given neighbor list format is valid.

    Parameters:
    -----------
    neighbor_list_format : _NeighborListFormat
        The neighbor list format to be validated.

    Returns:
    --------
    None

    Raises:
    -------
    ValueError
        If the neighbor list format is not one of the recognized formats.
    """
    if neighbor_list_format not in list(_NeighborListFormat):
        raise ValueError


def is_neighbor_list_sparse(
    neighbor_list_format: _NeighborListFormat,
) -> bool:
    r"""Determine if the given neighbor list format is sparse.

    Parameters:
    -----------
    neighbor_list_format : _NeighborListFormat
        The neighbor list format to be checked.

    Returns:
    --------
    bool
        True if the neighbor list format is sparse, False otherwise.
    """
    return neighbor_list_format in {
        _NeighborListFormat.ORDERED_SPARSE,
        _NeighborListFormat.SPARSE,
    }


def _is_space_valid(space: Tensor) -> Tensor:
    r"""Check if the given space tensor is valid.

    Parameters:
    -----------
    space : Tensor
        The space tensor to be validated. This tensor can have 0, 1, or 2 dimensions.

    Returns:
    --------
    Tensor
        A tensor containing a single boolean value indicating whether the space is valid.

    Raises:
    -------
    ValueError
        If the space tensor has more than 2 dimensions.
    """
    if space.ndim == 0 or space.ndim == 1:
        return torch.tensor([True], device=device)

    if space.ndim == 2:
        return torch.tensor([torch.all(torch.triu(space) == space)], device=device)

    return torch.tensor([False], device=device)


def neighbor_list_mask(neighbor: _NeighborList, mask_self: bool = False) -> Tensor:
    r"""Compute a mask for neighbor list.

    Parameters
    ----------
    neighbor : _NeighborList
      The input tensor to be masked.
    mask_self : bool


    Returns
    -------
    mask : Tensor
      The masked tensor.
    """
    if is_neighbor_list_sparse(neighbor.format):
        mask = neighbor.indexes[0] < len(neighbor.reference_positions)
        torch.set_printoptions(profile="full")
        if mask_self:
            mask = mask & (neighbor.indexes[0] != neighbor.indexes[1])

        return mask

    mask = neighbor.indexes < len(neighbor.indexes)

    if mask_self:
        N = len(neighbor.reference_positions)

        self_mask = neighbor.indexes != torch.arange(N).view(N, 1)

        mask = mask & self_mask

    return mask


def _neighboring_cell_lists(
    dimension: int,
) -> Generator[Tensor, None, None]:
    for index in torch.cartesian_prod(*[torch.arange(3) for _ in range(dimension)]):
        yield index - 1


def _normalize_cell_size(box: Tensor, cutoff: float) -> Tensor:
    r"""Normalize the cell size given the bounding box dimensions and a cutoff value.

    Parameters
    ----------
    box : Tensor
        A tensor representing the dimensions of the bounding box. The shape of
        the tensor can be:
        - Scalar (0D): representing a uniform box.
        - 1D: a vector representing the side lengths of the box.
        - 2D: a matrix representing the side lengths and angles between them.

    cutoff : float
        The cutoff value used for normalization.

    Returns
    -------
    Tensor
        A tensor representing the normalized cell size.

    Raises
    ------
    ValueError
        If `box` has more than 3 dimensions or if the shape is not supported.
    """
    if box.ndim == 0:
        return cutoff / box

    if box.ndim == 1:
        return cutoff / torch.min(box)

    if box.ndim == 2:
        if box.shape[0] == 1:
            return 1 / torch.floor(box[0, 0] / cutoff)

        if box.shape[0] == 2:
            xx = box[0, 0]
            yy = box[1, 1]
            xy = box[0, 1] / yy

            nx = xx / torch.sqrt(1 + xy**2)
            ny = yy

            nmin = torch.floor(torch.min(torch.tensor([nx, ny], device=device)) / cutoff)

            return 1 / torch.where(nmin == 0, 1, nmin)

        if box.shape[0] == 3:
            xx = box[0, 0]
            yy = box[1, 1]
            zz = box[2, 2]
            xy = box[0, 1] / yy
            xz = box[0, 2] / zz
            yz = box[1, 2] / zz

            nx = xx / torch.sqrt(1 + xy**2 + (xy * yz - xz) ** 2)
            ny = yy / torch.sqrt(1 + yz**2)
            nz = zz

            nmin = torch.floor(torch.min(torch.tensor([nx, ny, nz], device=device)) / cutoff)
            return 1 / torch.where(nmin == 0, 1, nmin)
        else:
            raise ValueError
    else:
        raise ValueError


def _particles_per_cell(
    positions: Tensor,
    size: Tensor,
    minimum_size: float,
) -> Tensor:
    r"""Computes the number of particles per cell given a defined cell size and minimum size.

    Parameters
    ----------
    positions : Tensor
        A tensor representing the positions of the particles in the system.
        The shape of the tensor is expected to be (n, d) where `n` is the number
        of particles and `d` is the dimensionality of the system.

    size : Tensor
        A tensor that defines the size of the space in each dimension.
        It should have the same shape as a single particle position.

    minimum_size : float
        A scalar that defines the minimum size of the cells.
        All cells will be at least this size in each dimension.

    Returns
    -------
    Tensor
        A tensor with the number of particles per cell. Each position in the tensor
        corresponds to a cell in the grid defined by the `size` and `minimum_size`
        parameters. The value at each position is the count of particles in that cell.
    """
    dim = positions.shape[1]

    size, unit_size, per_side, n = _cell_dimensions(dim, size, minimum_size)

    hash_multipliers = _hash_constants(dim, per_side).to(device=device)

    positions = positions.to(device=device)
    unit_size = unit_size.to(device=device)

    particle_index = torch.tensor(positions / unit_size, dtype=torch.int32, device=device)

    particle_hash = torch.sum(particle_index * hash_multipliers, dim=1)

    filling = _segment_sum(torch.ones_like(particle_hash).to(device=device), particle_hash, n)

    return filling


def cell_list(
    size: Tensor,
    minimum_unit_size: float,
    buffer_size_multiplier: float = 1.25,
) -> _CellListFunctionList:
    r"""Construct a cell list for neighbor searching in particle simulations.

    This function constructs a cell list used in particle simulations to accelerate the search for neighboring particles. Particle positions are grouped into cells based on their coordinates, and the cell list can be used to quickly find particles within a specified radius.

    Parameters
    ----------
    size : Tensor
        The size of the simulation box. If a 1-dimensional tensor is passed, it's reshaped to (1, -1).
    minimum_unit_size : float
        The minimum size of the simulation cells.
    buffer_size_multiplier : float, default=1.25
        A multiplier to determine the buffer size for each cell.

    Returns
    -------
    _CellListFunctionList
        An object containing `setup_fn` and `update_fn` functions to create and update the cell list.
    """
    if not isinstance(size, Tensor):
        size = torch.tensor(size, device=device, dtype=torch.float32)

    if size.ndim == 1:
        size = torch.reshape(size, [1, -1])

    def fn(
        positions: Tensor,
        excess: tuple[bool, int, Callable[..., _CellList]] | None = None,
        excess_buffer_size: int = 0,
        **kwargs,
    ) -> _CellList:
        r"""Create or update the cell list with particle positions and optional excess buffer.

        Parameters
        ----------
        positions : Tensor
            Positions of the particles.
        excess : tuple[bool, int, Callable[..., _CellList]] | None, default=None
            Information about excess buffer size.
        excess_buffer_size : int, default=0
            Size of the excess buffer.
        **kwargs : dict
            Additional parameters to store alongside the particle positions.

        Returns
        -------
        _CellList
            The constructed or updated cell list.
        """
        spatial_dimension = positions.shape[1]

        if spatial_dimension not in {2, 3}:
            raise ValueError

        _, unit_size, units_per_side, unit_count = _cell_dimensions(
            spatial_dimension,
            size,
            minimum_unit_size,
        )

        if excess is None:
            buffer_size = _estimate_cell_capacity(
                positions, size, unit_size, buffer_size_multiplier
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
                (unit_count * buffer_size,) + kwarg_shape,
                dtype=parameter.dtype,
                device=parameter.device,
            )

        positions = positions.to(device=device)
        unit_size = unit_size.to(device=device)

        hashes = torch.sum(
            (positions / unit_size).to(dtype=torch.int32, device=device)
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
            torch.max(_segment_sum(torch.ones_like(hashes).to(device=device), hashes, unit_count))
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
        r"""Setup the cell list with initial particle positions.

        Parameters
        ----------
        positions : Tensor
            Initial positions of the particles.
        excess_buffer_size : int, default=0
            Size of the excess buffer.
        **kwargs : dict
            Additional parameters to store alongside the particle positions.

        Returns
        -------
        _CellList
            The initialized cell list.
        """
        return fn(positions, excess_buffer_size=excess_buffer_size, **kwargs)

    def update_fn(
        positions: Tensor,
        buffer: int | _CellList,
        **kwargs,
    ) -> _CellList:
        r"""Update the cell list with new particle positions.

        Parameters
        ----------
        positions : Tensor
            New positions of the particles.
        buffer : int or _CellList
            Either an integer specifying the buffer size or an existing `_CellList` object.
        **kwargs : dict
            Additional parameters to store alongside the particle positions.

        Returns
        -------
        _CellList
            The updated cell list.
        """
        if isinstance(buffer, int):
            return fn(positions, (buffer, False, fn), **kwargs)

        return fn(
            positions,
            (buffer.size, buffer.exceeded_maximum_size, buffer.update_fn),
            **kwargs,
        )

    return _CellListFunctionList(setup_fn=setup_fn, update_fn=update_fn)


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
    r"""Creates a neighbor list function list based on the given parameters.

    Parameters
    ----------
    displacement_fn : Callable[[Tensor, Tensor], Tensor]
        A function that computes the displacement between two tensors.
    space : Tensor
        The space in which the particles are located.
    neighborhood_radius : float
        The radius within which neighbors are considered.
    maximum_distance : float, optional
        The maximum distance for considering neighbors (default is 0.0).
    buffer_size_multiplier : float, optional
        A multiplier for the buffer size (default is 1.25).
    disable_unit_list : bool, optional
        Whether to disable the unit list (default is False).
    mask_self : bool, optional
        Whether to mask self interactions (default is True).
    mask_fn : Optional[Callable[[Tensor], Tensor]], optional
        A function to mask certain interactions (default is None).
    normalized : bool, optional
        Whether the space is normalized (default is False).
    neighbor_list_format : _NeighborListFormat, optional
        The format of the neighbor list (default is _NeighborListFormat.DENSE).
    **_
        Additional keyword arguments.

    Returns
    -------
    _NeighborListFunctionList
        A function list for setting up and updating the neighbor list.
    """
    _is_neighbor_list_format_valid(neighbor_list_format)

    space = space.detach()

    cutoff = neighborhood_radius + maximum_distance

    squared_cutoff = cutoff**2

    squared_maximum_distance = (maximum_distance / 2.0) ** 2

    metric_sq = _to_square_metric_fn(displacement_fn)

    def _neighbor_candidate_fn(shape: tuple[int, ...]) -> Tensor:
        return torch.broadcast_to(
            torch.arange(shape[0], dtype=torch.int32)[None, :], (shape[0], shape[0])
        )

    def _cell_list_neighbor_candidate_fn(unit_indexes_buffer, shape) -> Tensor:
        n, spatial_dimension = shape

        indexes = unit_indexes_buffer

        unit_indexes = [indexes]

        for dindex in _neighboring_cell_lists(spatial_dimension):
            if torch.all(dindex == 0):
                continue

            unit_indexes += [_shift(indexes, dindex)]

        unit_indexes = torch.concatenate(unit_indexes, dim=-2)

        unit_indexes = unit_indexes[..., None, :, :]

        unit_indexes = torch.broadcast_to(
            unit_indexes, indexes.shape[:-1] + unit_indexes.shape[-2:]
        )

        def copy_values_from_cell(value, cell_value, cell_id):
            value = value.to(device=device)
            scatter_indices = torch.reshape(cell_id, (-1,)).to(device=device)

            cell_value = torch.reshape(cell_value, (-1,) + cell_value.shape[-2:]).to(device=device)

            value[scatter_indices] = cell_value

            return value

        neighbor_indexes = torch.zeros(
            (n + 1,) + unit_indexes.shape[-2:], dtype=torch.int32, device=device
        )

        neighbor_indexes = copy_values_from_cell(
            neighbor_indexes, unit_indexes, indexes
        )

        return neighbor_indexes[:-1, :, 0]

    def mask_self_fn(idx: Tensor) -> Tensor:
        return torch.where(
            idx
            == torch.reshape(
                torch.arange(idx.shape[0], dtype=torch.int32, device=device),
                (idx.shape[0], 1),
            ),
            idx.shape[0],
            idx,
        ).to(device=device)

    def prune_dense_neighbor_list(
        positions: Tensor, indexes: Tensor, **kwargs
    ) -> Tensor:
        positions = positions.to(device=device)
        indexes = indexes.to(device=device)

        displacement_fn = functools.partial(metric_sq, **kwargs)

        displacement_fn = _map_neighbor(displacement_fn)

        neighbor_positions = safe_index(positions, indexes)

        displacements = displacement_fn(positions, neighbor_positions)

        mask = (displacements < squared_cutoff) & (indexes < positions.shape[0])

        output_indexes = positions.shape[0] * torch.ones(
            indexes.shape, dtype=torch.int32, device=device
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
        distances = displacement_fn(
            safe_index(position, sender_idx), safe_index(position, receiver_idx)
        )

        mask = (distances < squared_cutoff) & (receiver_idx < position.shape[0])

        if neighbor_list_format is _NeighborListFormat.ORDERED_SPARSE:
            mask = mask & (receiver_idx < sender_idx)

        out_idx = position.shape[0] * torch.ones(receiver_idx.shape, dtype=torch.int32, device=device)

        cumsum = torch.cumsum(torch.flatten(mask), dim=0).to(device=device)
        index = torch.where(mask, cumsum - 1, len(receiver_idx) - 1).to(device=device)

        index = index.to(torch.int64)
        sender_idx = sender_idx.to(torch.int32)

        receiver_idx = out_idx.scatter(0, index, receiver_idx)
        sender_idx = out_idx.scatter(0, index, sender_idx)

        max_occupancy = cumsum[-1]

        return torch.stack((receiver_idx, sender_idx)), max_occupancy

    def _neighbors_fn(
        positions: Tensor, neighbors=None, extra_capacity: int = 0, **kwargs
    ) -> _NeighborList:
        def _fn(position_and_error, maximum_size=None):
            reference_positions, err = position_and_error

            n = reference_positions.shape[0]

            buffer_fn = None
            unit_list = None
            item_size = None

            if not disable_unit_list:
                if neighbors is None:
                    _space = kwargs.get("space", space)

                    item_size = cutoff
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
                        buffer_fn = cell_list(_space, item_size, buffer_size_multiplier)

                        unit_list = buffer_fn.setup_fn(
                            reference_positions, excess_buffer_size=extra_capacity
                        )

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

            if mask_self:
                indexes = mask_self_fn(indexes)

            if mask_fn is not None:
                indexes = mask_fn(indexes)

            if is_neighbor_list_sparse(neighbor_list_format):
                indexes, occupancy = prune_sparse_neighbor_list(
                    reference_positions, indexes, **kwargs
                )

            else:
                indexes, occupancy = prune_dense_neighbor_list(
                    reference_positions, indexes, **kwargs
                )

            if maximum_size is None:
                if not is_neighbor_list_sparse(neighbor_list_format):
                    _extra_capacity = extra_capacity
                else:
                    _extra_capacity = n * extra_capacity

                maximum_size = int(occupancy * buffer_size_multiplier + _extra_capacity)

                if maximum_size > indexes.shape[-1]:
                    maximum_size = indexes.shape[-1]

                if not is_neighbor_list_sparse(neighbor_list_format):
                    capacity_limit = n - 1 if mask_self else n

                elif neighbor_list_format == _NeighborListFormat.SPARSE:
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

        if "space" in kwargs and not disable_unit_list:
            if not normalized:
                raise ValueError

            current_unit_size = _cell_size(
                1.0,
                updated_neighbors.item_size,
            )

            updated_unit_size = _cell_size(
                1.0,
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

        displacement_fn = torch.vmap(displacement_fn)

        predicate = torch.any(
            displacement_fn(positions, updated_neighbors.reference_positions)
            > squared_maximum_distance
        )

        if predicate:
            return _fn((positions, updated_neighbors.partition_error))

        else:
            return neighbor_fn((positions, updated_neighbors.partition_error))

    def setup_fn(positions: Tensor, extra_capacity: int = 0, **kwargs):
        return _neighbors_fn(positions, extra_capacity=extra_capacity, **kwargs)

    def update_fn(positions: Tensor, neighbors, **kwargs):
        return _neighbors_fn(positions, neighbors, **kwargs)

    return _NeighborListFunctionList(setup_fn=setup_fn, update_fn=update_fn)


def partition(
    displacement_fn: Callable[[Tensor, Tensor], Tensor],
    space: Tensor,
    neighborhood_radius: float,
    *,
    maximum_distance: float = 0.0,
    buffer_size_multiplier: float = 1.25,
    disable_unit_list: bool = False,
    mask_self: bool = True,
    mask_fn: Optional[Callable[[Tensor], Tensor]] = None,
    normalized: bool = False,
    neighbor_list_format: _NeighborListFormat = _NeighborListFormat.DENSE,
    **_,
) -> Callable[[Tensor, Optional[_NeighborList], Optional[int]], _NeighborList]:
    r"""Partitions the space into a neighbor list based on the given parameters.

    Parameters
    ----------
    displacement_fn : Callable[[Tensor, Tensor], Tensor]
        A function that computes the displacement between two tensors.
    space : Tensor
        The space in which the particles are located.
    neighborhood_radius : float
        The radius within which neighbors are considered.
    maximum_distance : float, optional
        The maximum distance for considering neighbors (default is 0.0).
    buffer_size_multiplier : float, optional
        A multiplier for the buffer size (default is 1.25).
    disable_unit_list : bool, optional
        Whether to disable the unit list (default is False).
    mask_self : bool, optional
        Whether to mask self interactions (default is True).
    mask_fn : Optional[Callable[[Tensor], Tensor]], optional
        A function to mask certain interactions (default is None).
    normalized : bool, optional
        Whether the space is normalized (default is False).
    neighbor_list_format : _NeighborListFormat, optional
        The format of the neighbor list (default is _NeighborListFormat.DENSE).

    Returns
    -------
    Callable[[Tensor, Optional[_NeighborList], Optional[int]], _NeighborList]
        A function that generates a neighbor list based on the given parameters.
    """
    return neighbor_list(
        displacement_fn,
        space,
        neighborhood_radius,
        maximum_distance,
        buffer_size_multiplier,
        disable_unit_list,
        mask_self,
        mask_fn,
        normalized,
        neighbor_list_format,
    )
