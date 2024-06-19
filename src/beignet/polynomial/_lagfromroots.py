from .__from_roots import _from_roots
from ._lagline import lagline
from ._lagmul import lagmul


def lagfromroots(roots):
    return _from_roots(lagline, lagmul, roots)
