from .__fromroots import _fromroots
from ._hermeline import hermeline
from ._hermemul import hermemul


def hermefromroots(roots):
    return _fromroots(hermeline, hermemul, roots)
