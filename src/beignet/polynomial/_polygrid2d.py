from beignet.polynomial import _grid, polyval


def polygrid2d(x, y, c):
    return _grid(polyval, c, x, y)
