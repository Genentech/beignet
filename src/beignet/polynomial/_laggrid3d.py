from beignet.polynomial import _grid, lagval


def laggrid3d(x, y, z, c):
    return _grid(lagval, c, x, y, z)
