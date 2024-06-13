# import abc
# import numbers
# import os
# from abc import ABC
#
# import numpy
# import numpy.linalg
#
# import beignet
# from ._as_series import as_series
# from ._chebdomain import chebdomain
# from ._chebinterpolate import chebinterpolate
# from ._chebline import chebline
# from ._chebpow import chebpow
# from ._getdomain import getdomain
# from ._hermdomain import hermdomain
# from ._hermedomain import hermedomain
# from ._hermeline import hermeline
# from ._hermepow import hermepow
# from ._hermline import hermline
# from ._hermpow import hermpow
# from ._lagdomain import lagdomain
# from ._lagline import lagline
# from ._lagpow import lagpow
# from ._legdomain import legdomain
# from ._legline import legline
# from ._legpow import legpow
# from ._mapdomain import mapdomain
# from ._mapparms import mapparms
# from ._polydomain import polydomain
# from ._polyline import polyline
# from ._polypow import polypow
# from ._trimcoef import trimcoef
#
#
# class _Polynomial(ABC):
#     __hash__ = None
#
#     __array_ufunc__ = None
#
#     maxpower = 100
#
#     _superscript_mapping = str.maketrans(
#         {
#             "0": "⁰",
#             "1": "¹",
#             "2": "²",
#             "3": "³",
#             "4": "⁴",
#             "5": "⁵",
#             "6": "⁶",
#             "7": "⁷",
#             "8": "⁸",
#             "9": "⁹",
#         }
#     )
#     _subscript_mapping = str.maketrans(
#         {
#             "0": "₀",
#             "1": "₁",
#             "2": "₂",
#             "3": "₃",
#             "4": "₄",
#             "5": "₅",
#             "6": "₆",
#             "7": "₇",
#             "8": "₈",
#             "9": "₉",
#         }
#     )
#
#     _use_unicode = not os.name == "nt"
#
#     @property
#     def symbol(self):
#         return self._symbol
#
#     @property
#     @abc.abstractmethod
#     def domain(self):
#         pass
#
#     @property
#     @abc.abstractmethod
#     def window(self):
#         pass
#
#     @property
#     @abc.abstractmethod
#     def basis_name(self):
#         pass
#
#     @staticmethod
#     @abc.abstractmethod
#     def _add(c1, c2):
#         pass
#
#     @staticmethod
#     @abc.abstractmethod
#     def _sub(c1, c2):
#         pass
#
#     @staticmethod
#     @abc.abstractmethod
#     def _mul(c1, c2):
#         pass
#
#     @staticmethod
#     @abc.abstractmethod
#     def _div(c1, c2):
#         pass
#
#     @staticmethod
#     @abc.abstractmethod
#     def _pow(c, pow, maxpower=None):
#         pass
#
#     @staticmethod
#     @abc.abstractmethod
#     def _val(x, c):
#         pass
#
#     @staticmethod
#     @abc.abstractmethod
#     def _int(c, m, k, lbnd, scl):
#         pass
#
#     @staticmethod
#     @abc.abstractmethod
#     def _der(c, m, scl):
#         pass
#
#     @staticmethod
#     @abc.abstractmethod
#     def _fit(x, y, deg, rcond, full):
#         pass
#
#     @staticmethod
#     @abc.abstractmethod
#     def _line(off, scl):
#         pass
#
#     @staticmethod
#     @abc.abstractmethod
#     def _roots(c):
#         pass
#
#     @staticmethod
#     @abc.abstractmethod
#     def _fromroots(r):
#         pass
#
#     def has_samecoef(self, other):
#         if len(self.coef) != len(other.coef):
#             return False
#         elif not numpy.all(self.coef == other.coef):
#             return False
#         else:
#             return True
#
#     def has_samedomain(self, other):
#         return numpy.all(self.domain == other.domain)
#
#     def has_samewindow(self, other):
#         return numpy.all(self.window == other.window)
#
#     def has_sametype(self, other):
#         return isinstance(other, self.__class__)
#
#     def _get_coefficients(self, other):
#         if isinstance(other, _Polynomial):
#             if not isinstance(other, self.__class__):
#                 raise TypeError("Polynomial types differ")
#             elif not numpy.all(self.domain == other.domain):
#                 raise TypeError("Domains differ")
#             elif not numpy.all(self.window == other.window):
#                 raise TypeError("Windows differ")
#             elif self.symbol != other.symbol:
#                 raise ValueError("Polynomial symbols differ")
#             return other.coef
#         return other
#
#     def __init__(self, coef, domain=None, window=None, symbol="x"):
#         [coef] = as_series([coef], trim=False)
#         self.coef = coef
#
#         if domain is not None:
#             [domain] = as_series([domain], trim=False)
#             if len(domain) != 2:
#                 raise ValueError("Domain has wrong number of elements.")
#             self.domain = domain
#
#         if window is not None:
#             [window] = as_series([window], trim=False)
#             if len(window) != 2:
#                 raise ValueError("Window has wrong number of elements.")
#             self.window = window
#
#         try:
#             if not symbol.isidentifier():
#                 raise ValueError("Symbol string must be a valid Python identifier")
#
#         except AttributeError as error:
#             raise TypeError("Symbol must be a non-empty string") from error
#
#         self._symbol = symbol
#
#     def __repr__(self):
#         coef = repr(self.coef)[6:-1]
#         domain = repr(self.domain)[6:-1]
#         window = repr(self.window)[6:-1]
#         name = self.__class__.__name__
#         return (
#             f"{name}({coef}, domain={domain}, window={window}, "
#             f"symbol='{self.symbol}')"
#         )
#
#     def __getstate__(self):
#         ret = self.__dict__.copy()
#         ret["coef"] = self.coef.copy()
#         ret["domain"] = self.domain.copy()
#         ret["window"] = self.window.copy()
#         ret["symbol"] = self.symbol
#         return ret
#
#     def __setstate__(self, dict):
#         self.__dict__ = dict
#
#     def __call__(self, arg):
#         off, scl = mapparms(self.domain, self.window)
#         arg = off + scl * arg
#         return self._val(arg, self.coef)
#
#     def __iter__(self):
#         return iter(self.coef)
#
#     def __len__(self):
#         return len(self.coef)
#
#     def __neg__(self):
#         return self.__class__(-self.coef, self.domain, self.window, self.symbol)
#
#     def __pos__(self):
#         return self
#
#     def __add__(self, other):
#         othercoef = self._get_coefficients(other)
#         try:
#             coef = self._add(self.coef, othercoef)
#         except Exception:
#             return NotImplemented
#         return self.__class__(coef, self.domain, self.window, self.symbol)
#
#     def __sub__(self, other):
#         othercoef = self._get_coefficients(other)
#         try:
#             coef = self._sub(self.coef, othercoef)
#         except Exception:
#             return NotImplemented
#         return self.__class__(coef, self.domain, self.window, self.symbol)
#
#     def __mul__(self, other):
#         othercoef = self._get_coefficients(other)
#         try:
#             coef = self._mul(self.coef, othercoef)
#         except Exception:
#             return NotImplemented
#         return self.__class__(coef, self.domain, self.window, self.symbol)
#
#     def __truediv__(self, other):
#         if not isinstance(other, numbers.Number) or isinstance(other, bool):
#             raise TypeError(
#                 f"unsupported types for true division: "
#                 f"'{type(self)}', '{type(other)}'"
#             )
#         return self.__floordiv__(other)
#
#     def __floordiv__(self, other):
#         res = self.__divmod__(other)
#         if res is NotImplemented:
#             return res
#         return res[0]
#
#     def __mod__(self, other):
#         res = self.__divmod__(other)
#         if res is NotImplemented:
#             return res
#         return res[1]
#
#     def __divmod__(self, other):
#         othercoef = self._get_coefficients(other)
#         try:
#             quo, rem = self._div(self.coef, othercoef)
#         except ZeroDivisionError:
#             raise
#         except Exception:
#             return NotImplemented
#         quo = self.__class__(quo, self.domain, self.window, self.symbol)
#         rem = self.__class__(rem, self.domain, self.window, self.symbol)
#         return quo, rem
#
#     def __pow__(self, other):
#         coef = self._pow(self.coef, other, maxpower=self.maxpower)
#         res = self.__class__(coef, self.domain, self.window, self.symbol)
#         return res
#
#     def __radd__(self, other):
#         try:
#             coef = self._add(other, self.coef)
#         except Exception:
#             return NotImplemented
#         return self.__class__(coef, self.domain, self.window, self.symbol)
#
#     def __rsub__(self, other):
#         try:
#             coef = self._sub(other, self.coef)
#         except Exception:
#             return NotImplemented
#         return self.__class__(coef, self.domain, self.window, self.symbol)
#
#     def __rmul__(self, other):
#         try:
#             coef = self._mul(other, self.coef)
#         except Exception:
#             return NotImplemented
#         return self.__class__(coef, self.domain, self.window, self.symbol)
#
#     def __rdiv__(self, other):
#         return self.__rfloordiv__(other)
#
#     def __rtruediv__(self, other):
#         return NotImplemented
#
#     def __rfloordiv__(self, other):
#         res = self.__rdivmod__(other)
#         if res is NotImplemented:
#             return res
#         return res[0]
#
#     def __rmod__(self, other):
#         res = self.__rdivmod__(other)
#         if res is NotImplemented:
#             return res
#         return res[1]
#
#     def __rdivmod__(self, other):
#         try:
#             quo, rem = self._div(other, self.coef)
#         except ZeroDivisionError:
#             raise
#         except Exception:
#             return NotImplemented
#         quo = self.__class__(quo, self.domain, self.window, self.symbol)
#         rem = self.__class__(rem, self.domain, self.window, self.symbol)
#         return quo, rem
#
#     def __eq__(self, other):
#         res = (
#             isinstance(other, self.__class__)
#             and numpy.all(self.domain == other.domain)
#             and numpy.all(self.window == other.window)
#             and (self.coef.shape == other.coef.shape)
#             and numpy.all(self.coef == other.coef)
#             and (self.symbol == other.symbol)
#         )
#         return res
#
#     def __ne__(self, other):
#         return not self.__eq__(other)
#
#     def copy(self):
#         return self.__class__(self.coef, self.domain, self.window, self.symbol)
#
#     def degree(self):
#         return len(self) - 1
#
#     def cutdeg(self, deg):
#         return self.truncate(deg + 1)
#
#     def trim(self, tol=0):
#         coef = trimcoef(self.coef, tol)
#         return self.__class__(coef, self.domain, self.window, self.symbol)
#
#     def truncate(self, size):
#         isize = int(size)
#         if isize != size or isize < 1:
#             raise ValueError("size must be a positive integer")
#         if isize >= len(self.coef):
#             coef = self.coef
#         else:
#             coef = self.coef[:isize]
#         return self.__class__(coef, self.domain, self.window, self.symbol)
#
#     def convert(self, domain=None, kind=None, window=None):
#         if kind is None:
#             kind = self.__class__
#         if domain is None:
#             domain = kind.domain
#         if window is None:
#             window = kind.window
#         return self(kind.identity(domain, window=window, symbol=self.symbol))
#
#     def mapparms(self):
#         return mapparms(self.domain, self.window)
#
#     def integ(self, m=1, k=None, lbnd=None):
#         if k is None:
#             k = []
#
#         off, scl = self.mapparms()
#         if lbnd is None:
#             lbnd = 0
#         else:
#             lbnd = off + scl * lbnd
#         coef = self._int(self.coef, m, k, lbnd, 1.0 / scl)
#         return self.__class__(coef, self.domain, self.window, self.symbol)
#
#     def deriv(self, m=1):
#         off, scl = self.mapparms()
#         coef = self._der(self.coef, m, scl)
#         return self.__class__(coef, self.domain, self.window, self.symbol)
#
#     def roots(self):
#         roots = self._roots(self.coef)
#         return mapdomain(roots, self.window, self.domain)
#
#     def linspace(self, n=100, domain=None):
#         if domain is None:
#             domain = self.domain
#         x = numpy.linspace(domain[0], domain[1], n)
#         y = self(x)
#         return x, y
#
#     @classmethod
#     def fit(
#         cls,
#         x,
#         y,
#         deg,
#         domain=None,
#         rcond=None,
#         full=False,
#         w=None,
#         window=None,
#         symbol="x",
#     ):
#         if domain is None:
#             domain = getdomain(x)
#         elif isinstance(domain, list) and len(domain) == 0:
#             domain = cls.domain
#
#         if window is None:
#             window = cls.window
#
#         xnew = mapdomain(x, domain, window)
#         res = cls._fit(xnew, y, deg, w=w, rcond=rcond, full=full)
#         if full:
#             [coef, status] = res
#             return (cls(coef, domain=domain, window=window, symbol=symbol), status)
#         else:
#             coef = res
#             return cls(coef, domain=domain, window=window, symbol=symbol)
#
#     @classmethod
#     def fromroots(cls, roots, domain=None, window=None, symbol="x"):
#         if domain is None:
#             domain = []
#         [roots] = as_series([roots], trim=False)
#         if domain is None:
#             domain = getdomain(roots)
#         elif isinstance(domain, list) and len(domain) == 0:
#             domain = cls.domain
#
#         if window is None:
#             window = cls.window
#
#         deg = len(roots)
#         off, scl = mapparms(domain, window)
#         rnew = off + scl * roots
#         coef = cls._fromroots(rnew) / scl**deg
#         return cls(coef, domain=domain, window=window, symbol=symbol)
#
#     @classmethod
#     def identity(cls, domain=None, window=None, symbol="x"):
#         if domain is None:
#             domain = cls.domain
#         if window is None:
#             window = cls.window
#         off, scl = mapparms(window, domain)
#         coef = cls._line(off, scl)
#         return cls(coef, domain, window, symbol)
#
#     @classmethod
#     def basis(cls, deg, domain=None, window=None, symbol="x"):
#         if domain is None:
#             domain = cls.domain
#         if window is None:
#             window = cls.window
#         ideg = int(deg)
#
#         if ideg != deg or ideg < 0:
#             raise ValueError("deg must be non-negative integer")
#         return cls([0] * ideg + [1], domain, window, symbol)
#
#     @classmethod
#     def cast(cls, series, domain=None, window=None):
#         if domain is None:
#             domain = cls.domain
#         if window is None:
#             window = cls.window
#         return series.convert(domain, cls, window)
#
#
# class ChebyshevPolynomial(_Polynomial):
#     _add = staticmethod(beignet.add_chebyshev_series)
#     _sub = staticmethod(beignet.subtract_chebyshev_series)
#     _mul = staticmethod(beignet.multiply_chebyshev_series)
#     _div = staticmethod(beignet.divide_chebyshev_series)
#     _pow = staticmethod(chebpow)
#     _val = staticmethod(beignet.evaluate_chebyshev_series)
#     _int = staticmethod(beignet.integrate_chebyshev_series)
#     _der = staticmethod(beignet.differentiate_chebyshev_series)
#     _fit = staticmethod(beignet.fit_chebyshev_series)
#     _line = staticmethod(chebline)
#     _roots = staticmethod(beignet.chebyshev_series_roots)
#     _fromroots = staticmethod(beignet.chebyshev_series_from_roots)
#
#     @classmethod
#     def interpolate(cls, func, deg, domain=None, args=()):
#         if domain is None:
#             domain = cls.domain
#
#         def xfunc(x):
#             return func(mapdomain(x, cls.window, domain), *args)
#
#         coef = chebinterpolate(xfunc, deg)
#         return cls(coef, domain=domain)
#
#     domain = numpy.array(chebdomain)
#     window = numpy.array(chebdomain)
#     basis_name = "T"
#
#
# class Hermite(_Polynomial):
#     _add = staticmethod(beignet.add_physicists_hermite_series)
#     _sub = staticmethod(beignet.subtract_physicists_hermite_series)
#     _mul = staticmethod(beignet.multiply_physicists_hermite_series)
#     _div = staticmethod(beignet.divide_physicists_hermite_series)
#     _pow = staticmethod(hermpow)
#     _val = staticmethod(beignet.evaluate_physicists_hermite_series)
#     _int = staticmethod(beignet.integrate_physicists_hermite_series)
#     _der = staticmethod(beignet.differentiate_physicists_hermite_series)
#     _fit = staticmethod(beignet.fit_physicists_hermite_series)
#     _line = staticmethod(hermline)
#     _roots = staticmethod(beignet.physicists_hermite_series_roots)
#     _fromroots = staticmethod(beignet.physicists_hermite_series_from_roots)
#
#     domain = numpy.array(hermdomain)
#     window = numpy.array(hermdomain)
#     basis_name = "H"
#
#
# class HermiteE(_Polynomial):
#     _add = staticmethod(beignet.add_probabilists_hermite_series)
#     _sub = staticmethod(beignet.subtract_probabilists_hermite_series)
#     _mul = staticmethod(beignet.multiply_probabilists_hermite_series)
#     _div = staticmethod(beignet.divide_probabilists_hermite_series)
#     _pow = staticmethod(hermepow)
#     _val = staticmethod(beignet.evaluate_probabilists_hermite_series)
#     _int = staticmethod(beignet.integrate_probabilists_hermite_series)
#     _der = staticmethod(beignet.differentiate_probabilists_hermite_series)
#     _fit = staticmethod(beignet.fit_probabilists_hermite_series)
#     _line = staticmethod(hermeline)
#     _roots = staticmethod(beignet.probabilists_hermite_series_roots)
#     _fromroots = staticmethod(beignet.probabilists_hermite_series_from_roots)
#
#     domain = numpy.array(hermedomain)
#     window = numpy.array(hermedomain)
#     basis_name = "He"
#
#
# class LaguerrePolynomial(_Polynomial):
#     _add = staticmethod(beignet.add_laguerre_series)
#     _sub = staticmethod(beignet.subtract_laguerre_series)
#     _mul = staticmethod(beignet.multiply_laguerre_series)
#     _div = staticmethod(beignet.divide_laguerre_series)
#     _pow = staticmethod(lagpow)
#     _val = staticmethod(beignet.evaluate_laguerre_series)
#     _int = staticmethod(beignet.integrate_laguerre_series)
#     _der = staticmethod(beignet.differentiate_laguerre_series)
#     _fit = staticmethod(beignet.fit_laguerre_series)
#     _line = staticmethod(lagline)
#     _roots = staticmethod(beignet.laguerre_series_roots)
#     _fromroots = staticmethod(beignet.laguerre_series_from_roots)
#
#     domain = numpy.array(lagdomain)
#     window = numpy.array(lagdomain)
#     basis_name = "L"
#
#
# class LegendrePolynomial(_Polynomial):
#     _add = staticmethod(beignet.add_legendre_series)
#     _sub = staticmethod(beignet.subtract_legendre_series)
#     _mul = staticmethod(beignet.multiply_legendre_series)
#     _div = staticmethod(beignet.divide_legendre_series)
#     _pow = staticmethod(legpow)
#     _val = staticmethod(beignet.evaluate_legendre_series)
#     _int = staticmethod(beignet.integrate_legendre_series)
#     _der = staticmethod(beignet.differentiate_legendre_series)
#     _fit = staticmethod(beignet.fit_legendre_series)
#     _line = staticmethod(legline)
#     _roots = staticmethod(beignet.legendre_series_roots)
#     _fromroots = staticmethod(beignet.legendre_series_from_roots)
#
#     domain = numpy.array(legdomain)
#     window = numpy.array(legdomain)
#     basis_name = "P"
#
#
# class Polynomial(_Polynomial):
#     _add = staticmethod(beignet.add_power_series)
#     _sub = staticmethod(beignet.subtract_power_series)
#     _mul = staticmethod(beignet.multiply_power_series)
#     _div = staticmethod(beignet.divide_power_series)
#     _pow = staticmethod(polypow)
#     _val = staticmethod(beignet.evaluate_power_series)
#     _int = staticmethod(beignet.integrate_power_series)
#     _der = staticmethod(beignet.differentiate_power_series)
#     _fit = staticmethod(beignet.fit_power_series)
#     _line = staticmethod(polyline)
#     _roots = staticmethod(beignet.power_series_roots)
#     _fromroots = staticmethod(beignet.power_series_from_roots)
#
#     domain = numpy.array(polydomain)
#     window = numpy.array(polydomain)
#     basis_name = None
