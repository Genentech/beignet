from beignet.polynomial import _from_roots, hermline, hermmul


def hermfromroots(input):
    return _from_roots(hermline, hermmul, input)
