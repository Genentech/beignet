from .chebyshev import Chebyshev
from .hermite import Hermite
from .hermite_e import HermiteE
from .laguerre import Laguerre
from .legendre import Legendre
from .polynomial import Polynomial


def set_default_printstyle(style):
    if style not in ("unicode", "ascii"):
        raise ValueError(
            f"Unsupported format string '{style}'. Valid options are 'ascii' "
            f"and 'unicode'"
        )
    _use_unicode = True
    if style == "ascii":
        _use_unicode = False
    from ._polybase import ABCPolyBase

    ABCPolyBase._use_unicode = _use_unicode
