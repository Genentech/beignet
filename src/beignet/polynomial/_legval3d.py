from beignet.polynomial import _evaluate, legval


def legval3d(x, y, z, c):
    return _evaluate(legval, c, x, y, z)
