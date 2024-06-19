from beignet.polynomial import _evaluate, polyval


def polyval2d(x, y, c):
    return _evaluate(polyval, c, x, y)
