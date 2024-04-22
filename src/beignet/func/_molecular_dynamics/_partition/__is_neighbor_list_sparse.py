from .__neighbor_list_format import (
    _NeighborListFormat,
)


def _is_neighbor_list_sparse(
    neighbor_list_format: _NeighborListFormat,
) -> bool:
    return neighbor_list_format in {
        _NeighborListFormat.ORDERED_SPARSE,
        _NeighborListFormat.SPARSE,
    }
