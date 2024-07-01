from beignet.polynomial import lagline, lagmul
from beignet.polynomial.__from_roots import _from_roots


def lagfromroots(roots):
    return _from_roots(lagline, lagmul, roots)
