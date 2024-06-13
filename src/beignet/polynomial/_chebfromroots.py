from .__fromroots import _fromroots
from ._chebline import chebline
from ._chebmul import chebmul


def chebfromroots(roots):
    return _fromroots(chebline, chebmul, roots)
