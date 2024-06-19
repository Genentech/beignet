from beignet.polynomial import _evaluate, chebval


def chebval2d(x, y, c):
    return _evaluate(chebval, c, x, y)
