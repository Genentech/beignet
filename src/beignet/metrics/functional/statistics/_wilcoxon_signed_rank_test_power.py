"""wilcoxon signed rank test power functional metric."""

import beignet.statistics


def wilcoxon_signed_rank_test_power(*args, **kwargs):
    """
    Compute wilcoxon_signed_rank_test_power.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.wilcoxon_signed_rank_test_power(*args, **kwargs)
