"""kruskal wallis test sample size functional metric."""

import beignet.statistics


def kruskal_wallis_test_sample_size(*args, **kwargs):
    """
    Compute kruskal_wallis_test_sample_size.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.kruskal_wallis_test_sample_size(*args, **kwargs)
