"""Analysis of variance power functional metric."""

import beignet.statistics


def analysis_of_variance_power(*args, **kwargs):
    """
    Compute statistical power for ANOVA.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.analysis_of_variance_power(*args, **kwargs)
