from beignet.polynomial import _grid, hermeval


def hermegrid2d(x, y, c):
    return _grid(hermeval, c, x, y)
