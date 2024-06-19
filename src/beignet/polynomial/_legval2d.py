from beignet.polynomial import _evaluate, legval


def legval2d(x, y, c):
    return _evaluate(legval, c, x, y)
