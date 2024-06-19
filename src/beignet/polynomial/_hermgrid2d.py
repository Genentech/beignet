from beignet.polynomial import _grid, hermval


def hermgrid2d(x, y, c):
    return _grid(hermval, c, x, y)
