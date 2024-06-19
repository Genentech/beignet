from beignet.polynomial import _fit, polyvander


def polyfit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(polyvander, x, y, deg, rcond, full, w)
