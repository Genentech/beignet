from beignet.polynomial import _from_roots, legline, legmul


def legfromroots(roots):
    return _from_roots(legline, legmul, roots)
