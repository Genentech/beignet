from beignet.polynomial import _grid, legval


def leggrid3d(x, y, z, c):
    return _grid(legval, c, x, y, z)
