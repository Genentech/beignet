from beignet.polynomial import _from_roots, hermeline, hermemul


def hermefromroots(input):
    return _from_roots(hermeline, hermemul, input)
