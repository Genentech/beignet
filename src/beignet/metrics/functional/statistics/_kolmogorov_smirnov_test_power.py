"""kolmogorov smirnov test power functional metric."""

import beignet.statistics


def kolmogorov_smirnov_test_power(*args, **kwargs):
    """
    Compute kolmogorov_smirnov_test_power.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.kolmogorov_smirnov_test_power(*args, **kwargs)
