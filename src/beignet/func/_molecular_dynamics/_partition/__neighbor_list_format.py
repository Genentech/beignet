from enum import Enum


class _NeighborListFormat(Enum):
    DENSE = 0
    ORDERED_SPARSE = 1
    SPARSE = 2
