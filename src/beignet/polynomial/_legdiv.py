from beignet.polynomial import _div, legmul


def legdiv(c1, c2):
    return _div(legmul, c1, c2)
