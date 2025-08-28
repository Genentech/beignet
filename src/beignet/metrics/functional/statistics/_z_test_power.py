"""Z-test power functional metric."""

import beignet.statistics


def z_test_power(*args, **kwargs):
    """
    Compute statistical power for z-test.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.z_test_power(*args, **kwargs)
