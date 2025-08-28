"""correlation power functional metric."""

import beignet.statistics


def correlation_power(*args, **kwargs):
    """
    Compute correlation power.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.correlation_power(*args, **kwargs)
