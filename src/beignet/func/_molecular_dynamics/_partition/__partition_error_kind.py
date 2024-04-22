from enum import IntEnum


class _PartitionErrorKind(IntEnum):
    NONE = 0
    NEIGHBOR_LIST_OVERFLOW = 1 << 0
    CELL_LIST_OVERFLOW = 1 << 1
    CELL_SIZE_TOO_SMALL = 1 << 2
    MALFORMED_BOX = 1 << 3
