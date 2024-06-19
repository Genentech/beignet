from beignet.polynomial import _evaluate, hermeval


def hermeval3d(x, y, z, c):
    return _evaluate(hermeval, c, x, y, z)
