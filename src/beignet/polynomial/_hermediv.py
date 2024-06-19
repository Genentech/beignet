from beignet.polynomial import _div, hermemul


def hermediv(c1, c2):
    return _div(hermemul, c1, c2)
