from beignet.polynomial import _evaluate, hermval


def hermval2d(x, y, c):
    return _evaluate(hermval, c, x, y)
