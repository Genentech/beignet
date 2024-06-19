from .__from_roots import _from_roots
from ._chebline import chebline
from ._chebmul import chebmul


def chebfromroots(input):
    return _from_roots(chebline, chebmul, input)
