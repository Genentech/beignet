from beignet.polynomial import _pow, lagmul


def lagpow(c, pow, maxpower=16):
    return _pow(lagmul, c, pow, maxpower)
