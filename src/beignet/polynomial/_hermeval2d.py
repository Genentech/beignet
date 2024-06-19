from beignet.polynomial import _evaluate, hermeval


def hermeval2d(x, y, c):
    return _evaluate(hermeval, c, x, y)
