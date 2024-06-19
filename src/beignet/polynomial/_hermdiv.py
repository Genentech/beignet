from .__div import _div
from ._hermmul import hermmul


def hermdiv(c1, c2):
    return _div(hermmul, c1, c2)
