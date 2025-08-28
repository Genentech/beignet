"""cliffs delta functional metric."""

import beignet.statistics


def cliffs_delta(*args, **kwargs):
    """
    Compute cliffs_delta.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.cliffs_delta(*args, **kwargs)
