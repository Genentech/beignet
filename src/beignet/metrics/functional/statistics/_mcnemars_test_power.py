"""mcnemars test power functional metric."""

import beignet.statistics


def mcnemars_test_power(*args, **kwargs):
    """
    Compute mcnemars_test_power.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.mcnemars_test_power(*args, **kwargs)
