"""cohens kappa power functional metric."""

import beignet.statistics


def cohens_kappa_power(*args, **kwargs):
    """
    Compute cohens_kappa_power.

    This is a functional wrapper around the beignet.statistics implementation.

    Parameters and return values match the underlying statistics function.
    """
    return beignet.statistics.cohens_kappa_power(*args, **kwargs)
