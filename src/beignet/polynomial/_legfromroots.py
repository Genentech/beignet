from beignet.polynomial import legline, legmul
from beignet.polynomial.__from_roots import _from_roots


def legfromroots(roots):
    return _from_roots(legline, legmul, roots)
