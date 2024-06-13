from .__pow import _pow
from ._hermemul import hermemul


def hermepow(c, pow, maxpower=16):
    return _pow(hermemul, c, pow, maxpower)
