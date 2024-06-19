from beignet.polynomial import _from_roots, polyline, polymul


def polyfromroots(roots):
    return _from_roots(polyline, polymul, roots)
