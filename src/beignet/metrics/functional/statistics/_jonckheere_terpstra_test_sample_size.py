"""jonckheere terpstra test sample size functional metric."""

import beignet.statistics


def jonckheere_terpstra_test_sample_size(*args, **kwargs):
    """
    Compute jonckheere_terpstra_test_sample_size.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.jonckheere_terpstra_test_sample_size(*args, **kwargs)
