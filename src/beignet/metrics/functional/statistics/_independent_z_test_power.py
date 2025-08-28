"""independent z test power functional metric."""

import beignet.statistics


def independent_z_test_power(*args, **kwargs):
    """
    Compute independent z test power.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.independent_z_test_power(*args, **kwargs)
