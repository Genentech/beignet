"""partial eta squared functional metric."""

import beignet.statistics


def partial_eta_squared(*args, **kwargs):
    """
    Compute partial_eta_squared.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.partial_eta_squared(*args, **kwargs)
