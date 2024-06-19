from .__div import _div
from ._lagmul import lagmul


def lagdiv(c1, c2):
    return _div(lagmul, c1, c2)
