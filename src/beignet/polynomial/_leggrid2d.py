from beignet.polynomial import _grid, legval


def leggrid2d(x, y, c):
    return _grid(legval, c, x, y)
