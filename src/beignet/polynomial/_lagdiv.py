from beignet.polynomial import _div, lagmul


def lagdiv(c1, c2):
    return _div(lagmul, c1, c2)
