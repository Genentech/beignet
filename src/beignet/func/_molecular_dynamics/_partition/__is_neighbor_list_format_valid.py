from .__neighbor_list_format import (
    _NeighborListFormat,
)


def _is_neighbor_list_format_valid(neighbor_list_format: _NeighborListFormat):
    if neighbor_list_format not in list(_NeighborListFormat):
        raise ValueError
