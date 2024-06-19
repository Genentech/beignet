from beignet.polynomial import _evaluate, hermval


def hermval3d(x, y, z, c):
    return _evaluate(hermval, c, x, y, z)
