"""f test minimum detectable effect functional metric."""

import beignet.statistics


def f_test_minimum_detectable_effect(*args, **kwargs):
    """
    Compute f_test_minimum_detectable_effect.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.f_test_minimum_detectable_effect(*args, **kwargs)
