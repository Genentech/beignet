"""welch t test power functional metric."""

import beignet.statistics


def welch_t_test_power(*args, **kwargs):
    """
    Compute welch_t_test_power.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.welch_t_test_power(*args, **kwargs)
