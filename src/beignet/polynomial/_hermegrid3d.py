from beignet.polynomial import _grid, hermeval


def hermegrid3d(x, y, z, c):
    return _grid(hermeval, c, x, y, z)
