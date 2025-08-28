"""glass delta functional metric."""

import beignet.statistics


def glass_delta(*args, **kwargs):
    """
    Compute glass_delta.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.glass_delta(*args, **kwargs)
