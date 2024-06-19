from beignet.polynomial import _grid, chebval


def chebgrid3d(x, y, z, c):
    return _grid(chebval, c, x, y, z)
