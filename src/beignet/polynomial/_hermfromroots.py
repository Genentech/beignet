from .__fromroots import _fromroots
from ._hermline import hermline
from ._hermmul import hermmul


def hermfromroots(roots):
    return _fromroots(hermline, hermmul, roots)
