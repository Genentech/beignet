from .__from_roots import _from_roots
from ._polyline import polyline
from ._polymul import polymul


def polyfromroots(roots):
    return _from_roots(polyline, polymul, roots)
