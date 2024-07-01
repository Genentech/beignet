from beignet.polynomial import _from_roots, hermline, hermmul


def hermfromroots(roots):
    return _from_roots(hermline, hermmul, roots)
