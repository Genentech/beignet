from beignet.polynomial import _evaluate, polyval


def polyval3d(x, y, z, c):
    return _evaluate(polyval, c, x, y, z)
