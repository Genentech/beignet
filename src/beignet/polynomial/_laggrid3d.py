from .__gridnd import _gridnd
from ._lagval import lagval


def laggrid3d(x, y, z, c):
    return _gridnd(lagval, c, x, y, z)
