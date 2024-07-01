from beignet.polynomial import _from_roots, lagline, lagmul


def lagfromroots(roots):
    return _from_roots(lagline, lagmul, roots)
