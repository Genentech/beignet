from beignet.polynomial import _fit, chebvander


def chebfit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(chebvander, x, y, deg, rcond, full, w)
