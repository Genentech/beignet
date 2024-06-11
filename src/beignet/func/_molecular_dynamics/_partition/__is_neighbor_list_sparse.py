from .__neighbor_list_format import (
    _NeighborListFormat,
)


def _is_neighbor_list_sparse(
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
