from beignet.polynomial import _evaluate, chebval


def chebval3d(x, y, z, c):
    return _evaluate(chebval, c, x, y, z)
