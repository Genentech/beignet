from beignet.polynomial import hermline, hermmul
from beignet.polynomial.__from_roots import _from_roots


def hermfromroots(roots):
    return _from_roots(hermline, hermmul, roots)
