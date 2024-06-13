from .__pow import _pow
from ._hermmul import hermmul


def hermpow(c, pow, maxpower=16):
    return _pow(hermmul, c, pow, maxpower)
