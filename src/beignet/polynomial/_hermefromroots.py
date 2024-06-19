from .__from_roots import _from_roots
from ._hermeline import hermeline
from ._hermemul import hermemul


def hermefromroots(input):
    return _from_roots(hermeline, hermemul, input)
