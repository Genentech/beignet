from beignet.polynomial import _fit, legvander


def legfit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(legvander, x, y, deg, rcond, full, w)
