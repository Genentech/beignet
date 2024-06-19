from beignet.polynomial import _pow, legmul


def legpow(c, pow, maxpower=16):
    return _pow(legmul, c, pow, maxpower)
