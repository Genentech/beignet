"""intraclass correlation power functional metric."""

import beignet.statistics


def intraclass_correlation_power(*args, **kwargs):
    """
    Compute intraclass_correlation_power.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.intraclass_correlation_power(*args, **kwargs)
