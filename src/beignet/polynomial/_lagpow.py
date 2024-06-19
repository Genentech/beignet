from .__pow import _pow
from ._lagmul import lagmul


def lagpow(c, pow, maxpower=16):
    return _pow(lagmul, c, pow, maxpower)
