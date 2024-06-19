from beignet.polynomial import _div, hermmul


def hermdiv(c1, c2):
    return _div(hermmul, c1, c2)
