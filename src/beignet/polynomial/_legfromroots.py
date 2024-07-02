from .__from_roots import _from_roots
from ._legline import legline
from ._legmul import legmul


def legfromroots(roots):
    return _from_roots(legline, legmul, roots)
