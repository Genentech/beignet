"""wilcoxon signed rank test sample size functional metric."""

import beignet.statistics


def wilcoxon_signed_rank_test_sample_size(*args, **kwargs):
    """
    Compute wilcoxon_signed_rank_test_sample_size.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.wilcoxon_signed_rank_test_sample_size(*args, **kwargs)
