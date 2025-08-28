"""friedman test power functional metric."""

import beignet.statistics


def friedman_test_power(*args, **kwargs):
    """
    Compute friedman_test_power.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.friedman_test_power(*args, **kwargs)
