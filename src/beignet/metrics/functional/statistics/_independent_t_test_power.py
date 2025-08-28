"""Independent t-test power functional metric."""

import beignet.statistics


def independent_t_test_power(*args, **kwargs):
    """
    Compute statistical power for independent t-test.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.independent_t_test_power(*args, **kwargs)
