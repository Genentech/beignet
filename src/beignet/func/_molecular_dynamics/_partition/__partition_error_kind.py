from enum import IntEnum


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
