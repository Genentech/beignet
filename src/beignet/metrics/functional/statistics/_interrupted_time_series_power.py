"""interrupted time series power functional metric."""

import beignet.statistics


def interrupted_time_series_power(*args, **kwargs):
    """
    Compute interrupted_time_series_power.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.interrupted_time_series_power(*args, **kwargs)
