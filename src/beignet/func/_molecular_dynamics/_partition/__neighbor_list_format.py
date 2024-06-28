from enum import Enum


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
