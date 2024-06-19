from beignet.polynomial import _pow, hermmul


def hermpow(c, pow, maxpower=16):
    return _pow(hermmul, c, pow, maxpower)
