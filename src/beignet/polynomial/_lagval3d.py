from beignet.polynomial import _evaluate, lagval


def lagval3d(x, y, z, c):
    return _evaluate(lagval, c, x, y, z)
