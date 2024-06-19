from beignet.polynomial import _grid, chebval


def chebgrid2d(x, y, c):
    return _grid(chebval, c, x, y)
