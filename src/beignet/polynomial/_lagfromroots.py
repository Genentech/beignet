from .__fromroots import _fromroots
from ._lagline import lagline
from ._lagmul import lagmul


def lagfromroots(roots):
    return _fromroots(lagline, lagmul, roots)
