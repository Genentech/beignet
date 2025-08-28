"""paired z test sample size functional metric."""

import beignet.statistics


def paired_z_test_sample_size(*args, **kwargs):
    """
    Compute paired_z_test_sample_size.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.paired_z_test_sample_size(*args, **kwargs)
