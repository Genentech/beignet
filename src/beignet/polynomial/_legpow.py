from .__pow import _pow
from ._legmul import legmul


def legpow(c, pow, maxpower=16):
    return _pow(legmul, c, pow, maxpower)
