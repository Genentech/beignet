"""paired t test power functional metric."""

import beignet.statistics


def paired_t_test_power(*args, **kwargs):
    """
    Compute paired_t_test_power.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.paired_t_test_power(*args, **kwargs)
