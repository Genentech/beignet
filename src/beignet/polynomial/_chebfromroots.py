from beignet.polynomial import _from_roots, chebline, chebmul


def chebfromroots(input):
    return _from_roots(chebline, chebmul, input)
