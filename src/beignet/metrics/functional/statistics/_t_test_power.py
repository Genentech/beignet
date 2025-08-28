"""T-test power functional metric."""

import beignet.statistics


def t_test_power(*args, **kwargs):
    """
    Compute statistical power for t-test.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.t_test_power(*args, **kwargs)
