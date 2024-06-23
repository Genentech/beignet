"""
================
Chebyshev Series
================

This module provides a number of functions useful for dealing with Chebyshev series.

Constants
---------

.. autosummary::
   :toctree: generated/

   chebdomain
   chebzero
   chebone
   chebx

Arithmetic
----------

.. autosummary::
   :toctree: generated/

   chebadd
   chebsub
   chebmulx
   chebmul
   chebdiv
   chebpow
   chebval
   chebval2d
   chebval3d
   chebgrid2d
   chebgrid3d

Calculus
--------

.. autosummary::
   :toctree: generated/

   chebder
   chebint

Misc Functions
--------------

.. autosummary::
   :toctree: generated/

   chebfromroots
   chebroots
   chebvander
   chebvander2d
   chebvander3d
   chebgauss
   chebweight
   chebcompanion
   chebfit
   chebpts1
   chebpts2
   chebtrim
   chebline
   cheb2poly
   poly2cheb
   chebinterpolate

"""

import jax
import jax.numpy

from . import polyutils

__all__ = [
    "chebzero",
    "chebone",
    "chebx",
    "chebdomain",
    "chebline",
    "chebadd",
    "chebsub",
    "chebmulx",
    "chebmul",
    "chebdiv",
    "chebpow",
    "chebval",
    "chebder",
    "chebint",
    "cheb2poly",
    "poly2cheb",
    "chebfromroots",
    "chebvander",
    "chebfit",
    "chebtrim",
    "chebroots",
    "chebpts1",
    "chebpts2",
    "chebval2d",
    "chebval3d",
    "chebgrid2d",
    "chebgrid3d",
    "chebvander2d",
    "chebvander3d",
    "chebcompanion",
    "chebgauss",
    "chebweight",
    "chebinterpolate",
]

chebtrim = polyutils.trimcoef

#
# A collection of functions for manipulating z-series. These are private
# functions and do minimal error checking.
#


def _cseries_to_zseries(c):
    """Convert Chebyshev series to z-series.

    Convert a Chebyshev series to the equivalent z-series. The result is
    never an empty array. The dtype of the return is the same as that of
    the input. No checks are run on the arguments as this routine is for
    internal use.

    Parameters
    ----------
    c : 1-D ndarray
        Chebyshev coefficients, ordered from low to high

    Returns
    -------
    zs : 1-D ndarray
        Odd length symmetric z-series, ordered from  low to high.

    """
    n = c.size
    zs = jax.numpy.zeros(2 * n - 1, dtype=c.dtype)
    zs = zs.at[n - 1 :].set(c / 2)
    return zs + zs[::-1]


def _zseries_to_cseries(zs):
    """Convert z-series to a Chebyshev series.

    Convert a z series to the equivalent Chebyshev series. The result is
    never an empty array. The dtype of the return is the same as that of
    the input. No checks are run on the arguments as this routine is for
    internal use.

    Parameters
    ----------
    zs : 1-D ndarray
        Odd length symmetric z-series, ordered from  low to high.

    Returns
    -------
    c : 1-D ndarray
        Chebyshev coefficients, ordered from  low to high.

    """
    n = (zs.size + 1) // 2
    c = zs[n - 1 :].copy()
    c = c.at[1:n].multiply(2)
    return c


def _zseries_mul(z1, z2, mode="full"):
    """Multiply two z-series.

    Multiply two z-series to produce a z-series.

    Parameters
    ----------
    z1, z2 : 1-D ndarray
        The arrays must be 1-D but this is not checked.
    mode : {"full", "same"}
        If "full", output has shape (len(c1) + len(c2) - 1). If "same", output has shape
        max(len(c1), len(c2)), possibly truncating high order modes.

    Returns
    -------
    product : 1-D ndarray
        The product z-series.

    Notes
    -----
    This is simply convolution. If symmetric/anti-symmetric z-series are
    denoted by S/A then the following rules apply:

    S*S, A*A -> S
    S*A, A*S -> A

    """
    return jax.numpy.convolve(z1, z2, mode=mode)


#
# Chebyshev series functions
#


def poly2cheb(pol):
    """Convert a polynomial to a Chebyshev series.

    Convert an array representing the coefficients of a polynomial (relative
    to the "standard" basis) ordered from lowest degree to highest, to an
    array of the coefficients of the equivalent Chebyshev series, ordered
    from lowest to highest degree.

    Parameters
    ----------
    pol : array_like
        1-D array containing the polynomial coefficients

    Returns
    -------
    c : ndarray
        1-D array containing the coefficients of the equivalent Chebyshev
        series.

    See Also
    --------
    cheb2poly

    """
    pol = polyutils.as_series(pol)
    deg = len(pol) - 1

    res = jax.numpy.zeros_like(pol)

    def body(i, res):
        k = deg - i
        res = chebadd(chebmulx(res, mode="same"), pol[k])
        return res

    res = jax.lax.fori_loop(0, deg + 1, body, res)
    return res


def cheb2poly(c):
    """Convert a Chebyshev series to a polynomial.

    Convert an array representing the coefficients of a Chebyshev series,
    ordered from lowest degree to highest, to an array of the coefficients
    of the equivalent polynomial (relative to the "standard" basis) ordered
    from lowest to highest degree.

    Parameters
    ----------
    c : array_like
        1-D array containing the Chebyshev series coefficients, ordered
        from lowest order term to highest.

    Returns
    -------
    pol : ndarray
        1-D array containing the coefficients of the equivalent polynomial
        (relative to the "standard" basis) ordered from lowest order term
        to highest.

    See Also
    --------
    poly2cheb

    """
    from .polynomial import polyadd, polymulx, polysub

    c = polyutils.as_series(c)
    n = len(c)
    if n < 3:
        return c
    else:
        c0 = jax.numpy.zeros_like(c).at[0].set(c[-2])
        c1 = jax.numpy.zeros_like(c).at[0].set(c[-1])

        # i is the current degree of c1
        def body(k, c0c1):
            i = n - 1 - k
            c0, c1 = c0c1
            tmp = c0
            c0 = polysub(c[i - 2], c1)
            c1 = polyadd(tmp, polymulx(c1, "same") * 2)
            return c0, c1

        c0, c1 = jax.lax.fori_loop(0, n - 2, body, (c0, c1))

        return polyadd(c0, polymulx(c1, "same"))


chebdomain = jax.numpy.array([-1, 1])
"""Chebyshev default domain."""

chebzero = jax.numpy.array([0])
"""Chebyshev coefficients representing zero."""

chebone = jax.numpy.array([1])
"""Chebyshev coefficients representing one."""

chebx = jax.numpy.array([0, 1])
"""Chebyshev coefficients representing the identity x."""


def chebline(off, scl):
    """Chebyshev series whose graph is a straight line.

    Parameters
    ----------
    off, scl : scalars
        The specified line is given by ``off + scl*x``.

    Returns
    -------
    y : ndarray
        This module's representation of the Chebyshev series for
        ``off + scl*x``.

    See Also
    --------
    orthax.polynomial.polyline
    orthax.legendre.legline
    orthax.laguerre.lagline
    orthax.hermite.hermline
    orthax.hermite_e.hermeline

    Examples
    --------
    >>> import beignet.orthax.chebyshev as C
    >>> C.chebline(3,2)
    array([3, 2])
    >>> C.chebval(-3, C.chebline(3,2)) # should be -3
    -3.0

    """
    return jax.numpy.array([off, scl])


def chebfromroots(roots):
    """Generate a Chebyshev series with given roots.

    The function returns the coefficients of the polynomial

    .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),

    in Chebyshev form, where the `r_n` are the roots specified in `roots`.
    If a zero has multiplicity n, then it must appear in `roots` n times.
    For instance, if 2 is a root of multiplicity three and 3 is a root of
    multiplicity 2, then `roots` looks something like [2, 2, 2, 3, 3]. The
    roots can appear in any order.

    If the returned coefficients are `c`, then

    .. math:: p(x) = c_0 + c_1 * T_1(x) + ... +  c_n * T_n(x)

    The coefficient of the last term is not generally 1 for monic
    polynomials in Chebyshev form.

    Parameters
    ----------
    roots : array_like
        Sequence containing the roots.

    Returns
    -------
    out : ndarray
        1-D array of coefficients.  If all roots are real then `out` is a
        real array, if some of the roots are complex, then `out` is complex
        even if all the coefficients in the result are real (see Examples
        below).

    See Also
    --------
    orthax.polynomial.polyfromroots
    orthax.legendre.legfromroots
    orthax.laguerre.lagfromroots
    orthax.hermite.hermfromroots
    orthax.hermite_e.hermefromroots

    Examples
    --------
    >>> import beignet.orthax.chebyshev as C
    >>> C.chebfromroots((-1,0,1)) # x^3 - x relative to the standard basis
    array([ 0.  , -0.25,  0.  ,  0.25])
    >>> j = complex(0,1)
    >>> C.chebfromroots((-j,j)) # x^2 + 1 relative to the standard basis
    array([1.5+0.j, 0. +0.j, 0.5+0.j])

    """
    return polyutils._fromroots(chebline, chebmul, roots)


def chebadd(c1, c2):
    """Add one Chebyshev series to another.

    Returns the sum of two Chebyshev series `c1` + `c2`.  The arguments
    are sequences of coefficients ordered from lowest order term to
    highest, i.e., [1,2,3] represents the series ``T_0 + 2*T_1 + 3*T_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Chebyshev series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the Chebyshev series of their sum.

    See Also
    --------
    chebsub, chebmulx, chebmul, chebdiv, chebpow

    Notes
    -----
    Unlike multiplication, division, etc., the sum of two Chebyshev series
    is a Chebyshev series (without having to "reproject" the result onto
    the basis set) so addition, just like that of "standard" polynomials,
    is simply "component-wise."

    Examples
    --------
    >>> from beignet.orthax import chebyshev as C
    >>> c1 = (1,2,3)
    >>> c2 = (3,2,1)
    >>> C.chebadd(c1,c2)
    array([4., 4., 4.])

    """
    return polyutils._add(c1, c2)


def chebsub(c1, c2):
    """Subtract one Chebyshev series from another.

    Returns the difference of two Chebyshev series `c1` - `c2`.  The
    sequences of coefficients are from lowest order term to highest, i.e.,
    [1,2,3] represents the series ``T_0 + 2*T_1 + 3*T_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Chebyshev series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Of Chebyshev series coefficients representing their difference.

    See Also
    --------
    chebadd, chebmulx, chebmul, chebdiv, chebpow

    Notes
    -----
    Unlike multiplication, division, etc., the difference of two Chebyshev
    series is a Chebyshev series (without having to "reproject" the result
    onto the basis set) so subtraction, just like that of "standard"
    polynomials, is simply "component-wise."

    Examples
    --------
    >>> from beignet.orthax import chebyshev as C
    >>> c1 = (1,2,3)
    >>> c2 = (3,2,1)
    >>> C.chebsub(c1,c2)
    array([-2.,  0.,  2.])
    >>> C.chebsub(c2,c1) # -C.chebsub(c1,c2)
    array([ 2.,  0., -2.])

    """
    return polyutils._sub(c1, c2)


def chebmulx(c, mode="full"):
    """Multiply a Chebyshev series by x.

    Multiply the polynomial `c` by x, where x is the independent
    variable.


    Parameters
    ----------
    c : array_like
        1-D array of Chebyshev series coefficients ordered from low to
        high.
    mode : {"full", "same"}
        If "full", output has shape (len(c) + 1). If "same", output has shape
        (len(c)), possibly truncating high order modes.

    Returns
    -------
    out : ndarray
        Array representing the result of the multiplication.

    Examples
    --------
    >>> from beignet.orthax import chebyshev as C
    >>> C.chebmulx([1,2,3])
    array([1. , 2.5, 1. , 1.5])

    """
    c = polyutils.as_series(c)
    prd = jax.numpy.zeros(len(c) + 1, dtype=c.dtype)
    prd = prd.at[1].set(c[0])

    if len(c) > 1:
        tmp = c[1:] / 2
        prd = prd.at[2:].set(tmp)
        prd = prd.at[0:-2].add(tmp)

    if mode == "same":
        prd = prd[: len(c)]
    return prd


def chebmul(c1, c2, mode="full"):
    """Multiply one Chebyshev series by another.

    Returns the product of two Chebyshev series `c1` * `c2`.  The arguments
    are sequences of coefficients, from lowest order "term" to highest,
    e.g., [1,2,3] represents the series ``T_0 + 2*T_1 + 3*T_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Chebyshev series coefficients ordered from low to
        high.
    mode : {"full", "same"}
        If "full", output has shape (len(c1) + len(c2) - 1). If "same", output has shape
        max(len(c1), len(c2)), possibly truncating high order modes.

    Returns
    -------
    out : ndarray
        Of Chebyshev series coefficients representing their product.

    See Also
    --------
    chebadd, chebsub, chebmulx, chebdiv, chebpow

    Notes
    -----
    In general, the (polynomial) product of two C-series results in terms
    that are not in the Chebyshev polynomial basis set.  Thus, to express
    the product as a C-series, it is typically necessary to "reproject"
    the product onto said basis set, which typically produces
    "unintuitive live" (but correct) results; see Examples section below.

    Examples
    --------
    >>> from beignet.orthax import chebyshev as C
    >>> c1 = (1,2,3)
    >>> c2 = (3,2,1)
    >>> C.chebmul(c1,c2) # multiplication requires "reprojection"
    array([  6.5,  12. ,  12. ,   4. ,   1.5])

    """
    c1, c2 = polyutils.as_series(c1, c2)
    z1 = _cseries_to_zseries(c1)
    z2 = _cseries_to_zseries(c2)
    prd = _zseries_mul(z1, z2, mode=mode)
    ret = _zseries_to_cseries(prd)
    if mode == "same":
        ret = ret[: max(len(c1), len(c2))]

    return ret


def chebdiv(c1, c2):
    """Divide one Chebyshev series by another.

    Returns the quotient-with-remainder of two Chebyshev series
    `c1` / `c2`.  The arguments are sequences of coefficients from lowest
    order "term" to highest, e.g., [1,2,3] represents the series
    ``T_0 + 2*T_1 + 3*T_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Chebyshev series coefficients ordered from low to
        high.

    Returns
    -------
    [quo, rem] : ndarrays
        Of Chebyshev series coefficients representing the quotient and
        remainder.

    See Also
    --------
    chebadd, chebsub, chebmulx, chebmul, chebpow

    Notes
    -----
    In general, the (polynomial) division of one C-series by another
    results in quotient and remainder terms that are not in the Chebyshev
    polynomial basis set.  Thus, to express these results as C-series, it
    is typically necessary to "reproject" the results onto said basis
    set, which typically produces "unintuitive" (but correct) results;
    see Examples section below.

    Examples
    --------
    >>> from beignet.orthax import chebyshev as C
    >>> c1 = (1,2,3)
    >>> c2 = (3,2,1)
    >>> C.chebdiv(c1,c2) # quotient "intuitive," remainder not
    (array([3.]), array([-8., -4.]))
    >>> c2 = (0,1,2,3)
    >>> C.chebdiv(c2,c1) # neither "intuitive"
    (array([0., 2.]), array([-2., -4.]))

    """
    # note: not as efficient as np version that avoids loops, but can handle trailing
    # zeros in the series which np would otherwise trim.
    return polyutils._div(chebmul, c1, c2)


def chebpow(c, pow, maxpower=16):
    """Raise a Chebyshev series to a power.

    Returns the Chebyshev series `c` raised to the power `pow`. The
    argument `c` is a sequence of coefficients ordered from low to high.
    i.e., [1,2,3] is the series  ``T_0 + 2*T_1 + 3*T_2.``

    Parameters
    ----------
    c : array_like
        1-D array of Chebyshev series coefficients ordered from low to
        high.
    pow : integer
        Power to which the series will be raised
    maxpower : integer, optional
        Maximum power allowed. This is mainly to limit growth of the series
        to unmanageable size. Default is 16

    Returns
    -------
    coef : ndarray
        Chebyshev series of power.

    See Also
    --------
    chebadd, chebsub, chebmulx, chebmul, chebdiv

    Examples
    --------
    >>> from beignet.orthax import chebyshev as C
    >>> C.chebpow([1, 2, 3, 4], 2)
    array([15.5, 22. , 16. , ..., 12.5, 12. ,  8. ])

    """
    # note: this is more efficient than `pu._pow(chebmul, c1, c2)`, as it
    # avoids converting between z and c series repeatedly

    c = polyutils.as_series(c)
    power = int(pow)
    if power != pow or power < 0:
        raise ValueError("Power must be a non-negative integer.")
    elif maxpower is not None and power > maxpower:
        raise ValueError("Power is too large")
    elif power == 0:
        return jax.numpy.array([1], dtype=c.dtype)
    elif power == 1:
        return c
    else:
        prd = jax.numpy.zeros(len(c) * pow, dtype=c.dtype)
        prd = chebadd(prd, c)
        zs = _cseries_to_zseries(c)
        prd = _cseries_to_zseries(prd)

        # This can be made more efficient by using powers of two
        # in the usual way.
        def body(i, p):
            p = jax.numpy.convolve(p, zs, mode="same")
            return p

        prd = jax.lax.fori_loop(2, power + 1, body, prd)
        return _zseries_to_cseries(prd)


def chebder(c, m=1, scl=1, axis=0):
    """Differentiate a Chebyshev series.

    Returns the Chebyshev series coefficients `c` differentiated `m` times
    along `axis`.  At each iteration the result is multiplied by `scl` (the
    scaling factor is for use in a linear change of variable). The argument
    `c` is an array of coefficients from low to high degree along each
    axis, e.g., [1,2,3] represents the series ``1*T_0 + 2*T_1 + 3*T_2``
    while [[1,2],[1,2]] represents ``1*T_0(x)*T_0(y) + 1*T_1(x)*T_0(y) +
    2*T_0(x)*T_1(y) + 2*T_1(x)*T_1(y)`` if axis=0 is ``x`` and axis=1 is
    ``y``.

    Parameters
    ----------
    c : array_like
        Array of Chebyshev series coefficients. If c is multidimensional
        the different axis correspond to different variables with the
        degree in each axis given by the corresponding index.
    m : int, optional
        Number of derivatives taken, must be non-negative. (Default: 1)
    scl : scalar, optional
        Each differentiation is multiplied by `scl`.  The end result is
        multiplication by ``scl**m``.  This is for use in a linear change of
        variable. (Default: 1)
    axis : int, optional
        Axis over which the derivative is taken. (Default: 0).

    Returns
    -------
    der : ndarray
        Chebyshev series of the derivative.

    See Also
    --------
    chebint

    Notes
    -----
    In general, the result of differentiating a C-series needs to be
    "reprojected" onto the C-series basis set. Thus, typically, the
    result of this function is "unintuitive," albeit correct; see Examples
    section below.

    Examples
    --------
    >>> from beignet.orthax import chebyshev as C
    >>> c = (1,2,3,4)
    >>> C.chebder(c)
    array([14., 12., 24.])
    >>> C.chebder(c,3)
    array([96.])
    >>> C.chebder(c,scl=-1)
    array([-14., -12., -24.])
    >>> C.chebder(c,2,-1)
    array([12.,  96.])

    """
    if m < 0:
        raise ValueError("The order of derivation must be non-negative")

    c = polyutils.as_series(c)

    if m == 0:
        return c

    c = jax.numpy.moveaxis(c, axis, 0)
    n = len(c)
    if m >= n:
        c = jax.numpy.zeros_like(c[:1])
    else:
        # TODO: figure out how to get rid of this python loop
        for _ in range(m):
            n = n - 1
            c *= scl
            der = jax.numpy.empty((n,) + c.shape[1:], dtype=c.dtype)

            # TODO: can this be vectorized?
            def body(k, der_c, n=n):
                j = n - k
                der, c = der_c
                der = der.at[j - 1].set((2 * j) * c[j])
                c = c.at[j - 2].add((j * c[j]) / (j - 2))
                return der, c

            der, c = jax.lax.fori_loop(0, n - 2, body, (der, c))
            if n > 1:
                der = der.at[1].set(4 * c[2])
            der = der.at[0].set(c[1])
            c = der

    c = jax.numpy.moveaxis(c, 0, axis)
    return c


def chebint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    """Integrate a Chebyshev series.

    Returns the Chebyshev series coefficients `c` integrated `m` times from
    `lbnd` along `axis`. At each iteration the resulting series is
    **multiplied** by `scl` and an integration constant, `k`, is added.
    The scaling factor is for use in a linear change of variable.  ("Buyer
    beware": note that, depending on what one is doing, one may want `scl`
    to be the reciprocal of what one might expect; for more information,
    see the Notes section below.)  The argument `c` is an array of
    coefficients from low to high degree along each axis, e.g., [1,2,3]
    represents the series ``T_0 + 2*T_1 + 3*T_2`` while [[1,2],[1,2]]
    represents ``1*T_0(x)*T_0(y) + 1*T_1(x)*T_0(y) + 2*T_0(x)*T_1(y) +
    2*T_1(x)*T_1(y)`` if axis=0 is ``x`` and axis=1 is ``y``.

    Parameters
    ----------
    c : array_like
        Array of Chebyshev series coefficients. If c is multidimensional
        the different axis correspond to different variables with the
        degree in each axis given by the corresponding index.
    m : int, optional
        Order of integration, must be positive. (Default: 1)
    k : {[], list, scalar}, optional
        Integration constant(s).  The value of the first integral at zero
        is the first value in the list, the value of the second integral
        at zero is the second value, etc.  If ``k == []`` (the default),
        all constants are set to zero.  If ``m == 1``, a single scalar can
        be given instead of a list.
    lbnd : scalar, optional
        The lower bound of the integral. (Default: 0)
    scl : scalar, optional
        Following each integration the result is *multiplied* by `scl`
        before the integration constant is added. (Default: 1)
    axis : int, optional
        Axis over which the integral is taken. (Default: 0).

    Returns
    -------
    S : ndarray
        C-series coefficients of the integral.

    Raises
    ------
    ValueError
        If ``m < 1``, ``len(k) > m``, ``np.ndim(lbnd) != 0``, or
        ``np.ndim(scl) != 0``.

    See Also
    --------
    chebder

    Notes
    -----
    Note that the result of each integration is *multiplied* by `scl`.
    Why is this important to note?  Say one is making a linear change of
    variable :math:`u = ax + b` in an integral relative to `x`.  Then
    :math:`dx = du/a`, so one will need to set `scl` equal to
    :math:`1/a`- perhaps not what one would have first thought.

    Also note that, in general, the result of integrating a C-series needs
    to be "reprojected" onto the C-series basis set.  Thus, typically,
    the result of this function is "unintuitive," albeit correct; see
    Examples section below.

    Examples
    --------
    >>> from beignet.orthax import chebyshev as C
    >>> c = (1,2,3)
    >>> C.chebint(c)
    array([ 0.5, -0.5,  0.5,  0.5])
    >>> C.chebint(c,3)
    array([ 0.03125   , -0.1875    ,  0.04166667, -0.05208333,  0.01041667, # may vary
        0.00625   ])
    >>> C.chebint(c, k=3)
    array([ 3.5, -0.5,  0.5,  0.5])
    >>> C.chebint(c,lbnd=-2)
    array([ 8.5, -0.5,  0.5,  0.5])
    >>> C.chebint(c,scl=-2)
    array([-1.,  1., -1., -1.])

    """
    if k is None:
        k = []
    c = polyutils.as_series(c)
    lbnd, scl = map(jax.numpy.asarray, (lbnd, scl))

    if not jax.numpy.iterable(k):
        k = [k]
    if len(k) > m:
        raise ValueError("Too many integration constants")
    if jax.numpy.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if jax.numpy.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")

    if m == 0:
        return c

    c = jax.numpy.moveaxis(c, axis, 0)
    k = jax.numpy.array(list(k) + [0] * (m - len(k)), ndmin=1)

    # TODO: figure out how to get rid of this python loop
    for i in range(m):
        n = len(c)
        c *= scl
        tmp = jax.numpy.empty((n + 1,) + c.shape[1:], dtype=c.dtype)
        tmp = tmp.at[0].set(c[0] * 0)
        tmp = tmp.at[1].set(c[0])
        if n > 1:
            tmp = tmp.at[2].set(c[1] / 4)
        j = jax.numpy.arange(2, n)
        tmp = tmp.at[j + 1].set((c[j].T / (2 * (j + 1))).T)
        tmp = tmp.at[j - 1].add(-(c[j].T / (2 * (j - 1))).T)
        tmp = tmp.at[0].add(k[i] - chebval(lbnd, tmp))
        c = tmp
    c = jax.numpy.moveaxis(c, 0, axis)
    return c


def chebval(x, c, tensor=True):
    """Evaluate a Chebyshev series at points x.

    If `c` is of length `n + 1`, this function returns the value:

    .. math:: p(x) = c_0 * T_0(x) + c_1 * T_1(x) + ... + c_n * T_n(x)

    The parameter `x` is converted to an array only if it is a tuple or a
    list, otherwise it is treated as a scalar. In either case, either `x`
    or its elements must support multiplication and addition both with
    themselves and with the elements of `c`.

    If `c` is a 1-D array, then `p(x)` will have the same shape as `x`.  If
    `c` is multidimensional, then the shape of the result depends on the
    value of `tensor`. If `tensor` is true the shape will be c.shape[1:] +
    x.shape. If `tensor` is false the shape will be c.shape[1:]. Note that
    scalars have shape (,).

    Trailing zeros in the coefficients will be used in the evaluation, so
    they should be avoided if efficiency is a concern.

    Parameters
    ----------
    x : array_like, compatible object
        If `x` is a list or tuple, it is converted to an ndarray, otherwise
        it is left unchanged and treated as a scalar. In either case, `x`
        or its elements must support addition and multiplication with
        themselves and with the elements of `c`.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree n are contained in c[n]. If `c` is multidimensional the
        remaining indices enumerate multiple polynomials. In the two
        dimensional case the coefficients may be thought of as stored in
        the columns of `c`.
    tensor : boolean, optional
        If True, the shape of the coefficient array is extended with ones
        on the right, one for each dimension of `x`. Scalars have dimension 0
        for this action. The result is that every column of coefficients in
        `c` is evaluated for every element of `x`. If False, `x` is broadcast
        over the columns of `c` for the evaluation.  This keyword is useful
        when `c` is multidimensional. The default value is True.

    Returns
    -------
    values : ndarray, algebra_like
        The shape of the return value is described above.

    See Also
    --------
    chebval2d, chebgrid2d, chebval3d, chebgrid3d

    Notes
    -----
    The evaluation uses Clenshaw recursion, aka synthetic division.

    """
    c = polyutils.as_series(c)
    x = jax.numpy.asarray(x)
    if tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)

    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        x2 = 2 * x
        c0 = c[-2] * jax.numpy.ones_like(x)
        c1 = c[-1] * jax.numpy.ones_like(x)

        def body(i, val):
            c0, c1 = val
            tmp = c0
            c0 = c[-i] - c1
            c1 = tmp + c1 * x2
            return c0, c1

        c0, c1 = jax.lax.fori_loop(3, len(c) + 1, body, (c0, c1))

    return c0 + c1 * x


def chebval2d(x, y, c):
    r"""Evaluate a 2-D Chebyshev series at points (x, y).

    This function returns the values:

    .. math:: p(x,y) = \\sum_{i,j} c_{i,j} * T_i(x) * T_j(y)

    The parameters `x` and `y` are converted to arrays only if they are
    tuples or a lists, otherwise they are treated as a scalars and they
    must have the same shape after conversion. In either case, either `x`
    and `y` or their elements must support multiplication and addition both
    with themselves and with the elements of `c`.

    If `c` is a 1-D array a one is implicitly appended to its shape to make
    it 2-D. The shape of the result will be c.shape[2:] + x.shape.

    Parameters
    ----------
    x, y : array_like, compatible objects
        The two dimensional series is evaluated at the points `(x, y)`,
        where `x` and `y` must have the same shape. If `x` or `y` is a list
        or tuple, it is first converted to an ndarray, otherwise it is left
        unchanged and if it isn't an ndarray it is treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficient of the term
        of multi-degree i,j is contained in ``c[i,j]``. If `c` has
        dimension greater than 2 the remaining indices enumerate multiple
        sets of coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional Chebyshev series at points formed
        from pairs of corresponding values from `x` and `y`.

    See Also
    --------
    chebval, chebgrid2d, chebval3d, chebgrid3d

    """
    return polyutils._valnd(chebval, c, x, y)


def chebgrid2d(x, y, c):
    r"""Evaluate a 2-D Chebyshev series on the Cartesian product of x and y.

    This function returns the values:

    .. math:: p(a,b) = \\sum_{i,j} c_{i,j} * T_i(a) * T_j(b),

    where the points `(a, b)` consist of all pairs formed by taking
    `a` from `x` and `b` from `y`. The resulting points form a grid with
    `x` in the first dimension and `y` in the second.

    The parameters `x` and `y` are converted to arrays only if they are
    tuples or a lists, otherwise they are treated as a scalars. In either
    case, either `x` and `y` or their elements must support multiplication
    and addition both with themselves and with the elements of `c`.

    If `c` has fewer than two dimensions, ones are implicitly appended to
    its shape to make it 2-D. The shape of the result will be c.shape[2:] +
    x.shape + y.shape.

    Parameters
    ----------
    x, y : array_like, compatible objects
        The two dimensional series is evaluated at the points in the
        Cartesian product of `x` and `y`.  If `x` or `y` is a list or
        tuple, it is first converted to an ndarray, otherwise it is left
        unchanged and, if it isn't an ndarray, it is treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficient of the term of
        multi-degree i,j is contained in `c[i,j]`. If `c` has dimension
        greater than two the remaining indices enumerate multiple sets of
        coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional Chebyshev series at points in the
        Cartesian product of `x` and `y`.

    See Also
    --------
    chebval, chebval2d, chebval3d, chebgrid3d

    """
    return polyutils._gridnd(chebval, c, x, y)


def chebval3d(x, y, z, c):
    r"""Evaluate a 3-D Chebyshev series at points (x, y, z).

    This function returns the values:

    .. math:: p(x,y,z) = \\sum_{i,j,k} c_{i,j,k} * T_i(x) * T_j(y) * T_k(z)

    The parameters `x`, `y`, and `z` are converted to arrays only if
    they are tuples or a lists, otherwise they are treated as a scalars and
    they must have the same shape after conversion. In either case, either
    `x`, `y`, and `z` or their elements must support multiplication and
    addition both with themselves and with the elements of `c`.

    If `c` has fewer than 3 dimensions, ones are implicitly appended to its
    shape to make it 3-D. The shape of the result will be c.shape[3:] +
    x.shape.

    Parameters
    ----------
    x, y, z : array_like, compatible object
        The three dimensional series is evaluated at the points
        `(x, y, z)`, where `x`, `y`, and `z` must have the same shape.  If
        any of `x`, `y`, or `z` is a list or tuple, it is first converted
        to an ndarray, otherwise it is left unchanged and if it isn't an
        ndarray it is  treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficient of the term of
        multi-degree i,j,k is contained in ``c[i,j,k]``. If `c` has dimension
        greater than 3 the remaining indices enumerate multiple sets of
        coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the multidimensional polynomial on points formed with
        triples of corresponding values from `x`, `y`, and `z`.

    See Also
    --------
    chebval, chebval2d, chebgrid2d, chebgrid3d

    """
    return polyutils._valnd(chebval, c, x, y, z)


def chebgrid3d(x, y, z, c):
    r"""Evaluate a 3-D Chebyshev series on the Cartesian product of x, y, and z.

    This function returns the values:

    .. math:: p(a,b,c) = \\sum_{i,j,k} c_{i,j,k} * T_i(a) * T_j(b) * T_k(c)

    where the points `(a, b, c)` consist of all triples formed by taking
    `a` from `x`, `b` from `y`, and `c` from `z`. The resulting points form
    a grid with `x` in the first dimension, `y` in the second, and `z` in
    the third.

    The parameters `x`, `y`, and `z` are converted to arrays only if they
    are tuples or a lists, otherwise they are treated as a scalars. In
    either case, either `x`, `y`, and `z` or their elements must support
    multiplication and addition both with themselves and with the elements
    of `c`.

    If `c` has fewer than three dimensions, ones are implicitly appended to
    its shape to make it 3-D. The shape of the result will be c.shape[3:] +
    x.shape + y.shape + z.shape.

    Parameters
    ----------
    x, y, z : array_like, compatible objects
        The three dimensional series is evaluated at the points in the
        Cartesian product of `x`, `y`, and `z`.  If `x`,`y`, or `z` is a
        list or tuple, it is first converted to an ndarray, otherwise it is
        left unchanged and, if it isn't an ndarray, it is treated as a
        scalar.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree i,j are contained in ``c[i,j]``. If `c` has dimension
        greater than two the remaining indices enumerate multiple sets of
        coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional polynomial at points in the Cartesian
        product of `x` and `y`.

    See Also
    --------
    chebval, chebval2d, chebgrid2d, chebval3d

    """
    return polyutils._gridnd(chebval, c, x, y, z)


def chebvander(x, deg):
    """Pseudo-Vandermonde matrix of given degree.

    Returns the pseudo-Vandermonde matrix of degree `deg` and sample points
    `x`. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., i] = T_i(x),

    where `0 <= i <= deg`. The leading indices of `V` index the elements of
    `x` and the last index is the degree of the Chebyshev polynomial.

    If `c` is a 1-D array of coefficients of length `n + 1` and `V` is the
    matrix ``V = chebvander(x, n)``, then ``np.dot(V, c)`` and
    ``chebval(x, c)`` are the same up to roundoff.  This equivalence is
    useful both for least squares fitting and for the evaluation of a large
    number of Chebyshev series of the same degree and sample points.

    Parameters
    ----------
    x : array_like
        Array of points. The dtype is converted to float64 or complex128
        depending on whether any of the elements are complex. If `x` is
        scalar it is converted to a 1-D array.
    deg : int
        Degree of the resulting matrix.

    Returns
    -------
    vander : ndarray
        The pseudo Vandermonde matrix. The shape of the returned matrix is
        ``x.shape + (deg + 1,)``, where The last index is the degree of the
        corresponding Chebyshev polynomial.  The dtype will be the same as
        the converted `x`.

    """
    if deg < 0:
        raise ValueError("deg must be non-negative")

    x = jax.numpy.array(x, ndmin=1)
    dims = (deg + 1,) + x.shape
    dtyp = jax.numpy.promote_types(x.dtype, jax.numpy.array(0.0).dtype)
    x = x.astype(dtyp)
    v = jax.numpy.empty(dims, dtype=dtyp)
    v = v.at[0].set(jax.numpy.ones_like(x))
    # Use forward recursion to generate the entries.
    if deg > 0:
        v = v.at[1].set(x)
        x2 = 2 * x

        def body(i, v):
            return v.at[i].set(v[i - 1] * x2 - v[i - 2])

        v = jax.lax.fori_loop(2, deg + 1, body, v)

    return jax.numpy.moveaxis(v, 0, -1)


def chebvander2d(x, y, deg):
    r"""Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points `(x, y)`. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (deg[1] + 1)*i + j] = T_i(x) * T_j(y),

    where `0 <= i <= deg[0]` and `0 <= j <= deg[1]`. The leading indices of
    `V` index the points `(x, y)` and the last index encodes the degrees of
    the Chebyshev polynomials.

    If ``V = chebvander2d(x, y, [xdeg, ydeg])``, then the columns of `V`
    correspond to the elements of a 2-D coefficient array `c` of shape
    (xdeg + 1, ydeg + 1) in the order

    .. math:: c_{00}, c_{01}, c_{02} ... , c_{10}, c_{11}, c_{12} ...

    and ``np.dot(V, c.flat)`` and ``chebval2d(x, y, c)`` will be the same
    up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 2-D Chebyshev
    series of the same degrees and sample points.

    Parameters
    ----------
    x, y : array_like
        Arrays of point coordinates, all of the same shape. The dtypes
        will be converted to either float64 or complex128 depending on
        whether any of the elements are complex. Scalars are converted to
        1-D arrays.
    deg : list of ints
        List of maximum degrees of the form [x_deg, y_deg].

    Returns
    -------
    vander2d : ndarray
        The shape of the returned matrix is ``x.shape + (order,)``, where
        :math:`order = (deg[0]+1)*(deg[1]+1)`.  The dtype will be the same
        as the converted `x` and `y`.

    See Also
    --------
    chebvander, chebvander3d, chebval2d, chebval3d

    """
    return polyutils._vander_nd_flat((chebvander, chebvander), (x, y), deg)


def chebvander3d(x, y, z, deg):
    r"""Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points `(x, y, z)`. If `l, m, n` are the given degrees in `x, y, z`,
    then The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = T_i(x)*T_j(y)*T_k(z),

    where `0 <= i <= l`, `0 <= j <= m`, and `0 <= j <= n`.  The leading
    indices of `V` index the points `(x, y, z)` and the last index encodes
    the degrees of the Chebyshev polynomials.

    If ``V = chebvander3d(x, y, z, [xdeg, ydeg, zdeg])``, then the columns
    of `V` correspond to the elements of a 3-D coefficient array `c` of
    shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order

    .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...

    and ``np.dot(V, c.flat)`` and ``chebval3d(x, y, z, c)`` will be the
    same up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 3-D Chebyshev
    series of the same degrees and sample points.

    Parameters
    ----------
    x, y, z : array_like
        Arrays of point coordinates, all of the same shape. The dtypes will
        be converted to either float64 or complex128 depending on whether
        any of the elements are complex. Scalars are converted to 1-D
        arrays.
    deg : list of ints
        List of maximum degrees of the form [x_deg, y_deg, z_deg].

    Returns
    -------
    vander3d : ndarray
        The shape of the returned matrix is ``x.shape + (order,)``, where
        :math:`order = (deg[0]+1)*(deg[1]+1)*(deg[2]+1)`.  The dtype will
        be the same as the converted `x`, `y`, and `z`.

    See Also
    --------
    chebvander, chebvander3d, chebval2d, chebval3d

    """
    return polyutils._vander_nd_flat(
        (chebvander, chebvander, chebvander), (x, y, z), deg
    )


def chebfit(x, y, deg, rcond=None, full=False, w=None):
    r"""Least squares fit of Chebyshev series to data.

    Return the coefficients of a Chebyshev series of degree `deg` that is the
    least squares fit to the data values `y` given at points `x`. If `y` is
    1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple
    fits are done, one for each column of `y`, and the resulting
    coefficients are stored in the corresponding columns of a 2-D return.
    The fitted polynomial(s) are in the form

    .. math::  p(x) = c_0 + c_1 * T_1(x) + ... + c_n * T_n(x),

    where `n` is `deg`.

    Parameters
    ----------
    x : array_like, shape (M,)
        x-coordinates of the M sample points ``(x[i], y[i])``.
    y : array_like, shape (M,) or (M, K)
        y-coordinates of the sample points. Several data sets of sample
        points sharing the same x-coordinates can be fitted at once by
        passing in a 2D-array that contains one dataset per column.
    deg : int or 1-D array_like
        Degree(s) of the fitting polynomials. If `deg` is a single integer,
        all terms up to and including the `deg`'th term are included in the
        fit. A tuple of integers specifying the degrees of the terms to include
        may be used instead.
    rcond : float, optional
        Relative condition number of the fit. Singular values smaller than
        this relative to the largest singular value will be ignored. The
        default value is len(x)*eps, where eps is the relative precision of
        the float type, about 2e-16 in most cases.
    full : bool, optional
        Switch determining nature of return value. When it is False (the
        default) just the coefficients are returned, when True diagnostic
        information from the singular value decomposition is also returned.
    w : array_like, shape (`M`,), optional
        Weights. If not None, the weight ``w[i]`` applies to the unsquared
        residual ``y[i] - y_hat[i]`` at ``x[i]``. Ideally the weights are
        chosen so that the errors of the products ``w[i]*y[i]`` all have the
        same variance.  When using inverse-variance weighting, use
        ``w[i] = 1/sigma(y[i])``.  The default value is None.

    Returns
    -------
    coef : ndarray, shape (M,) or (M, K)
        Chebyshev coefficients ordered from low to high. If `y` was 2-D,
        the coefficients for the data in column k  of `y` are in column
        `k`.

    [residuals, rank, singular_values, rcond] : list
        These values are only returned if ``full == True``

        - residuals -- sum of squared residuals of the least squares fit
        - rank -- the numerical rank of the scaled Vandermonde matrix
        - singular_values -- singular values of the scaled Vandermonde matrix
        - rcond -- value of `rcond`.

        For more details, see `jax.numpy.linalg.lstsq`.

    See Also
    --------
    orthax.polynomial.polyfit
    orthax.legendre.legfit
    orthax.laguerre.lagfit
    orthax.hermite.hermfit
    orthax.hermite_e.hermefit
    chebval : Evaluates a Chebyshev series.
    chebvander : Vandermonde matrix of Chebyshev series.
    chebweight : Chebyshev weight function.
    jax.numpy.linalg.lstsq : Computes a least-squares fit from the matrix.

    Notes
    -----
    The solution is the coefficients of the Chebyshev series `p` that
    minimizes the sum of the weighted squared errors

    .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,

    where :math:`w_j` are the weights. This problem is solved by setting up
    as the (typically) overdetermined matrix equation

    .. math:: V(x) * c = w * y,

    where `V` is the weighted pseudo Vandermonde matrix of `x`, `c` are the
    coefficients to be solved for, `w` are the weights, and `y` are the
    observed values.  This equation is then solved using the singular value
    decomposition of `V`.

    If some of the singular values of `V` are so small that they are
    neglected, then a `RankWarning` will be issued. This means that the
    coefficient values may be poorly determined. Using a lower order fit
    will usually get rid of the warning.  The `rcond` parameter can also be
    set to a value smaller than its default, but the resulting fit may be
    spurious and have large contributions from roundoff error.

    Fits using Chebyshev series are usually better conditioned than fits
    using power series, but much can depend on the distribution of the
    sample points and the smoothness of the data. If the quality of the fit
    is inadequate splines may be a good alternative.

    """
    return polyutils._fit(chebvander, x, y, deg, rcond, full, w)


def chebcompanion(c):
    """Return the scaled companion matrix of c.

    The basis polynomials are scaled so that the companion matrix is
    symmetric when `c` is a Chebyshev basis polynomial. This provides
    better eigenvalue estimates than the unscaled case and for basis
    polynomials the eigenvalues are guaranteed to be real if
    `jax.numpy.linalg.eigvalsh` is used to obtain them.

    Parameters
    ----------
    c : array_like
        1-D array of Chebyshev series coefficients ordered from low to high
        degree.

    Returns
    -------
    mat : ndarray
        Scaled companion matrix of dimensions (deg, deg).

    """
    c = polyutils.as_series(c)
    if len(c) < 2:
        raise ValueError("Series must have maximum degree of at least 1.")
    if len(c) == 2:
        return jax.numpy.array([[-c[0] / c[1]]])

    n = len(c) - 1
    mat = jax.numpy.zeros((n, n), dtype=c.dtype)
    scl = jax.numpy.ones(n).at[1:].set(jax.numpy.sqrt(0.5))
    shp = mat.shape
    mat = mat.flatten()
    mat = mat.at[1 :: n + 1].set(
        jax.numpy.full(n - 1, 1 / 2).at[0].set(jax.numpy.sqrt(0.5))
    )
    mat = mat.at[n :: n + 1].set(
        jax.numpy.full(n - 1, 1 / 2).at[0].set(jax.numpy.sqrt(0.5))
    )
    mat = mat.reshape(shp)
    mat = mat.at[:, -1].add(-(c[:-1] / c[-1]) * (scl / scl[-1]) * 0.5)
    return mat


def chebroots(c):
    r"""Compute the roots of a Chebyshev series.

    Return the roots (a.k.a. "zeros") of the polynomial

    .. math:: p(x) = \\sum_i c[i] * T_i(x).

    Parameters
    ----------
    c : 1-D array_like
        1-D array of coefficients.

    Returns
    -------
    out : ndarray
        Array of the roots of the series. If all the roots are real,
        then `out` is also real, otherwise it is complex.

    See Also
    --------
    orthax.polynomial.polyroots
    orthax.legendre.legroots
    orthax.laguerre.lagroots
    orthax.hermite.hermroots
    orthax.hermite_e.hermeroots

    Notes
    -----
    The root estimates are obtained as the eigenvalues of the companion
    matrix, Roots far from the origin of the complex plane may have large
    errors due to the numerical instability of the series for such
    values. Roots with multiplicity greater than 1 will also show larger
    errors as the value of the series near such points is relatively
    insensitive to errors in the roots. Isolated roots near the origin can
    be improved by a few iterations of Newton's method.

    The Chebyshev series basis polynomials aren't powers of `x` so the
    results of this function may seem unintuitive.

    Examples
    --------
    >>> import beignet.orthax.chebyshev as cheb
    >>> cheb.chebroots((-1, 1,-1, 1)) # T3 - T2 + T1 - T0 has real roots
    array([ -5.00000000e-01,   2.60860684e-17,   1.00000000e+00]) # may vary

    """
    c = polyutils.as_series(c)
    if len(c) <= 1:
        return jax.numpy.array([], dtype=c.dtype)
    if len(c) == 2:
        return jax.numpy.array([-c[0] / c[1]])

    # rotated companion matrix reduces error
    m = chebcompanion(c)[::-1, ::-1]
    r = jax.numpy.linalg.eigvals(m)
    r = jax.numpy.sort(r)
    return r


def chebinterpolate(func, deg, args=()):
    """Interpolate a function at the Chebyshev points of the first kind.

    Returns the Chebyshev series that interpolates `func` at the Chebyshev
    points of the first kind in the interval [-1, 1]. The interpolating
    series tends to a minmax approximation to `func` with increasing `deg`
    if the function is continuous in the interval.

    Parameters
    ----------
    func : function
        The function to be approximated. It must be a function of a single
        variable of the form ``f(x, a, b, c...)``, where ``a, b, c...`` are
        extra arguments passed in the `args` parameter.
    deg : int
        Degree of the interpolating polynomial
    args : tuple, optional
        Extra arguments to be used in the function call. Default is no extra
        arguments.

    Returns
    -------
    coef : ndarray, shape (deg + 1,)
        Chebyshev coefficients of the interpolating series ordered from low to
        high.

    Examples
    --------
    >>> import beignet.orthax.chebyshev as C
    >>> C.chebinterpolate(lambda x: np.tanh(x) + 0.5, 8)
    array([  5.00000000e-01,   8.11675684e-01,  -9.86864911e-17,
            -5.42457905e-02,  -2.71387850e-16,   4.51658839e-03,
             2.46716228e-17,  -3.79694221e-04,  -3.26899002e-16])

    Notes
    -----
    The Chebyshev polynomials used in the interpolation are orthogonal when
    sampled at the Chebyshev points of the first kind. If it is desired to
    constrain some of the coefficients they can simply be set to the desired
    value after the interpolation, no new interpolation or fit is needed. This
    is especially useful if it is known a-priori that some of coefficients are
    zero. For instance, if the function is even then the coefficients of the
    terms of odd degree in the result can be set to zero.

    """
    _deg = int(deg)
    if _deg != deg:
        raise ValueError("deg must be integer")
    if _deg < 0:
        raise ValueError("expected deg >= 0")

    order = _deg + 1
    xcheb = chebpts1(order)
    yfunc = func(xcheb, *args)
    m = chebvander(xcheb, _deg)
    c = jax.numpy.dot(m.T, yfunc)
    c = c.at[0].divide(order)
    c = c.at[1:].divide(0.5 * order)

    return c


def chebgauss(deg):
    r"""Gauss-Chebyshev quadrature.

    Computes the sample points and weights for Gauss-Chebyshev quadrature.
    These sample points and weights will correctly integrate polynomials of
    degree :math:`2*deg - 1` or less over the interval :math:`[-1, 1]` with
    the weight function :math:`f(x) = 1/\\sqrt{1 - x^2}`.

    Parameters
    ----------
    deg : int
        Number of sample points and weights. It must be >= 1.

    Returns
    -------
    x : ndarray
        1-D ndarray containing the sample points.
    y : ndarray
        1-D ndarray containing the weights.

    Notes
    -----
    The results have only been tested up to degree 100, higher degrees may
    be problematic. For Gauss-Chebyshev there are closed form solutions for
    the sample points and weights. If n = `deg`, then

    .. math:: x_i = \\cos(\\pi (2 i - 1) / (2 n))

    .. math:: w_i = \\pi / n

    """
    deg = int(deg)
    if deg <= 0:
        raise ValueError("deg must be a positive integer")

    x = jax.numpy.cos(jax.numpy.pi * jax.numpy.arange(1, 2 * deg, 2) / (2.0 * deg))
    w = jax.numpy.ones(deg) * (jax.numpy.pi / deg)

    return x, w


def chebweight(x):
    r"""The weight function of the Chebyshev polynomials.

    The weight function is :math:`1/\\sqrt{1 - x^2}` and the interval of
    integration is :math:`[-1, 1]`. The Chebyshev polynomials are
    orthogonal, but not normalized, with respect to this weight function.

    Parameters
    ----------
    x : array_like
       Values at which the weight function will be computed.

    Returns
    -------
    w : ndarray
       The weight function at `x`.

    """
    x = jax.numpy.asarray(x)
    w = 1.0 / (jax.numpy.sqrt(1.0 + x) * jax.numpy.sqrt(1.0 - x))
    return w


def chebpts1(npts):
    """Chebyshev points of the first kind.

    The Chebyshev points of the first kind are the points ``cos(x)``,
    where ``x = [pi*(k + .5)/npts for k in range(npts)]``.

    Parameters
    ----------
    npts : int
        Number of sample points desired.

    Returns
    -------
    pts : ndarray
        The Chebyshev points of the first kind.

    See Also
    --------
    chebpts2

    """
    _npts = int(npts)
    if _npts != npts:
        raise ValueError("npts must be integer")
    if _npts < 1:
        raise ValueError("npts must be >= 1")

    x = 0.5 * jax.numpy.pi / _npts * jax.numpy.arange(-_npts + 1, _npts + 1, 2)
    return jax.numpy.sin(x)


def chebpts2(npts):
    """Chebyshev points of the second kind.

    The Chebyshev points of the second kind are the points ``cos(x)``,
    where ``x = [pi*k/(npts - 1) for k in range(npts)]`` sorted in ascending
    order.

    Parameters
    ----------
    npts : int
        Number of sample points desired.

    Returns
    -------
    pts : ndarray
        The Chebyshev points of the second kind.

    """
    _npts = int(npts)
    if _npts != npts:
        raise ValueError("npts must be integer")
    if _npts < 2:
        raise ValueError("npts must be >= 2")

    x = jax.numpy.linspace(-jax.numpy.pi, 0, _npts)
    return jax.numpy.cos(x)
