from beignet.polynomial import _grid, polyval


def polygrid3d(x, y, z, c):
    return _grid(polyval, c, x, y, z)
