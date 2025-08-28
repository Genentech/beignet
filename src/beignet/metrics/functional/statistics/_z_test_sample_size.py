"""z test sample size functional metric."""

import beignet.statistics


def z_test_sample_size(*args, **kwargs):
    """
    Compute z test sample size.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.z_test_sample_size(*args, **kwargs)
