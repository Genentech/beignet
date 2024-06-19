from beignet.polynomial import _pow, hermemul


def hermepow(c, pow, maxpower=16):
    return _pow(hermemul, c, pow, maxpower)
