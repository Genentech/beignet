from beignet.polynomial import _evaluate, lagval


def lagval2d(x, y, c):
    return _evaluate(lagval, c, x, y)
