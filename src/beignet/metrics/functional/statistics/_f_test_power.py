"""f test power functional metric."""

import beignet.statistics


def f_test_power(*args, **kwargs):
    """
    Compute f test power.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.f_test_power(*args, **kwargs)
