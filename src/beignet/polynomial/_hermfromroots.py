from .__from_roots import _from_roots
from ._hermline import hermline
from ._multiply_physicists_hermite_series import multiply_physicists_hermite_series


def hermfromroots(input):
    return _from_roots(hermline, multiply_physicists_hermite_series, input)
