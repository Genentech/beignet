from beignet.polynomial import _grid, hermval


def hermgrid3d(x, y, z, c):
    return _grid(hermval, c, x, y, z)
