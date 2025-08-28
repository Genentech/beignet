"""kruskal wallis test power functional metric."""

import beignet.statistics


def kruskal_wallis_test_power(*args, **kwargs):
    """
    Compute kruskal_wallis_test_power.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.kruskal_wallis_test_power(*args, **kwargs)
