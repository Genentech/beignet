from beignet.polynomial import _grid, lagval


def laggrid2d(x, y, c):
    return _grid(lagval, c, x, y)
