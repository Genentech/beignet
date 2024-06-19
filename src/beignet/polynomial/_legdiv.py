from .__div import _div
from ._legmul import legmul


def legdiv(c1, c2):
    return _div(legmul, c1, c2)
