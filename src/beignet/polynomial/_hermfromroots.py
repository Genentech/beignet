from .__from_roots import _from_roots
from ._hermline import hermline
from ._hermmul import hermmul


def hermfromroots(input):
    return _from_roots(hermline, hermmul, input)
