from .__fromroots import _fromroots
from ._polyline import polyline
from ._polymul import polymul


def polyfromroots(roots):
    return _fromroots(polyline, polymul, roots)
