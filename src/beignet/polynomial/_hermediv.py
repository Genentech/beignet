from .__div import _div
from ._hermemul import hermemul


def hermediv(c1, c2):
    return _div(hermemul, c1, c2)
