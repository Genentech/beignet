from .__fit import _fit
from ._legvander import legvander_vandermonde_1d


def legfit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(legvander_vandermonde_1d, x, y, deg, rcond, full, w)
