from .__from_roots import _from_roots
from ._hermeline import hermeline
from ._multiply_probabilists_hermite_series import multiply_probabilists_hermite_series


def hermefromroots(input):
    return _from_roots(hermeline, multiply_probabilists_hermite_series, input)
