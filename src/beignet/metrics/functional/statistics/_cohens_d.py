"""Cohen's d effect size functional metric."""

import beignet.statistics


def cohens_d(*args, **kwargs):
    """
    Compute Cohen's d effect size between two groups.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.cohens_d(*args, **kwargs)
