from .__neighbor_list_format import (
    _NeighborListFormat,
)


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
