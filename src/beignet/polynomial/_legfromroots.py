from .__fromroots import _fromroots
from ._legline import legline
from ._legmul import legmul


def legfromroots(roots):
    return _fromroots(legline, legmul, roots)
