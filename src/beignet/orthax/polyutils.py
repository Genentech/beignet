"""
=================
Utility functions
=================

Functions
---------

.. autosummary::
   :toctree: generated/

   as_series    convert array_likes into 1-D arrays of common type.
   trimseq      remove trailing zeros.
   trimcoef     remove small trailing coefficients.
   getdomain    return the domain appropriate for a given set of abscissae.
   mapdomain    maps points between domains.
   mapparms     parameters of the linear map between domains.

"""

import functools
import operator

import jax
import jax.numpy
import numpy

__all__ = [
    "as_series",
    "trimseq",
    "trimcoef",
    "getdomain",
    "mapdomain",
    "mapparms",
]


def trimseq(seq):
    """Remove small Poly series coefficients.

    Parameters
    ----------
    seq : sequence
        Sequence of Poly series coefficients.

    Returns
    -------
    series : sequence
        Subsequence with trailing zeros removed. If the resulting sequence
        would be empty, return the first element. The returned sequence may
        or may not be a view.

    Notes
    -----
    Not compatible with JAX transformations due to non-static output size.

    """
    if len(seq) == 0:
        return seq
    else:
        for i in range(len(seq) - 1, -1, -1):
            if seq[i] != 0:
                break
        return seq[: i + 1]


def as_series(*arrs, trim=False):
    """Return arguments as a list of 1-d arrays.

    The returned list contains array(s) of dtype double, complex double, or
    object.  A 1-d argument of shape ``(N,)`` is parsed into ``N`` arrays of
    size one; a 2-d argument of shape ``(M,N)`` is parsed into ``M`` arrays
    of size ``N`` (i.e., is "parsed by row"); and a higher dimensional array
    raises a Value Error if it is not first reshaped into either a 1-d or 2-d
    array.

    Parameters
    ----------
    arrs : array_like
        1- or 2-d array_like
    trim : boolean, optional
        When True, trailing zeros are removed from the inputs.
        When False, the inputs are passed through intact.

    Returns
    -------
    a1, a2,... : 1-D arrays
        A copy of the input data as 1-d arrays.

    """
    arrays = tuple(jax.numpy.array(a, ndmin=1) for a in arrs)
    if trim:
        arrays = tuple(trimseq(a) for a in arrays)
    arrays = jax._src.numpy.util.promote_dtypes_inexact(*arrays)
    if len(arrays) == 1:
        return arrays[0]
    return tuple(arrays)


def trimcoef(c, tol=0):
    """
    Remove "small" "trailing" coefficients from a polynomial.

    "Small" means "small in absolute value" and is controlled by the
    parameter `tol`; "trailing" means highest order coefficient(s), e.g., in
    ``[0, 1, 1, 0, 0]`` (which represents ``0 + x + x**2 + 0*x**3 + 0*x**4``)
    both the 3-rd and 4-th order coefficients would be "trimmed."

    Parameters
    ----------
    c : array_like
        1-d array of coefficients, ordered from lowest order to highest.
    tol : number, optional
        Trailing (i.e., highest order) elements with absolute value less
        than or equal to `tol` (default value is zero) are removed.

    Returns
    -------
    trimmed : ndarray
        1-d array with trailing zeros removed.  If the resulting series
        would be empty, a series containing a single zero is returned.

    Raises
    ------
    ValueError
        If `tol` < 0

    Notes
    -----
    Not compatible with JAX transformations due to non-static output size.

    See Also
    --------
    trimseq

    Examples
    --------
    >>> from beignet.orthax import polyutils as pu
    >>> pu.trimcoef((0,0,3,0,5,0,0))
    array([0.,  0.,  3.,  0.,  5.])
    >>> pu.trimcoef((0,0,1e-3,0,1e-5,0,0),1e-3) # item == tol is trimmed
    array([0.])
    >>> i = complex(0,1) # works for complex
    >>> pu.trimcoef((3e-4,1e-3*(1-i),5e-4,2e-5*(1+i)), 1e-3)
    array([0.0003+0.j   , 0.001 -0.001j])

    """
    if tol < 0:
        raise ValueError("tol must be non-negative")

    c = as_series(c)
    [ind] = jax.numpy.nonzero(jax.numpy.abs(c) > tol)
    if len(ind) == 0:
        return c[:1] * 0
    else:
        return c[: ind[-1] + 1].copy()


@jax.jit
def getdomain(x):
    """
    Return a domain suitable for given abscissae.

    Find a domain suitable for a polynomial or Chebyshev series
    defined at the values supplied.

    Parameters
    ----------
    x : array_like
        1-d array of abscissae whose domain will be determined.

    Returns
    -------
    domain : ndarray
        1-d array containing two values.  If the inputs are complex, then
        the two returned points are the lower left and upper right corners
        of the smallest rectangle (aligned with the axes) in the complex
        plane containing the points `x`. If the inputs are real, then the
        two points are the ends of the smallest interval containing the
        points `x`.

    See Also
    --------
    mapparms, mapdomain

    Examples
    --------
    >>> from beignet.orthax import polyutils as pu
    >>> points = numpy.arange(4)**2 - 5; points
    array([-5, -4, -1,  4])
    >>> pu.getdomain(points)
    array([-5.,  4.])
    >>> c = numpy.exp(complex(0,1)*numpy.pi*numpy.arange(12)/6) # unit circle
    >>> pu.getdomain(c)
    array([-1.-1.j,  1.+1.j])

    """
    x = jax.numpy.asarray(x)
    if jax.numpy.iscomplexobj(x):
        rmin, rmax = x.real.min(), x.real.max()
        imin, imax = x.imag.min(), x.imag.max()
        return jax.numpy.array(((rmin + 1j * imin), (rmax + 1j * imax)))
    else:
        return jax.numpy.array((x.min(), x.max()))


@jax.jit
def mapparms(old, new):
    """
    Linear map parameters between domains.

    Return the parameters of the linear map ``offset + scale*x`` that maps
    `old` to `new` such that ``old[i] -> new[i]``, ``i = 0, 1``.

    Parameters
    ----------
    old, new : array_like
        Domains. Each domain must (successfully) convert to a 1-d array
        containing precisely two values.

    Returns
    -------
    offset, scale : scalars
        The map ``L(x) = offset + scale*x`` maps the first domain to the
        second.

    See Also
    --------
    getdomain, mapdomain

    Notes
    -----
    Also works for complex numbers, and thus can be used to calculate the
    parameters required to map any line in the complex plane to any other
    line therein.

    Examples
    --------
    >>> from beignet.orthax import polyutils as pu
    >>> pu.mapparms((-1,1),(-1,1))
    (0.0, 1.0)
    >>> pu.mapparms((1,-1),(-1,1))
    (-0.0, -1.0)
    >>> i = complex(0,1)
    >>> pu.mapparms((-i,-1),(1,i))
    ((1+1j), (1-0j))

    """
    oldlen = old[1] - old[0]
    newlen = new[1] - new[0]
    off = (old[1] * new[0] - old[0] * new[1]) / oldlen
    scl = newlen / oldlen
    return off, scl


@jax.jit
def mapdomain(x, old, new):
    r"""
    Apply linear map to input points.

    The linear map ``offset + scale*x`` that maps the domain `old` to
    the domain `new` is applied to the points `x`.

    Parameters
    ----------
    x : array_like
        Points to be mapped. If `x` is a subtype of ndarray the subtype
        will be preserved.
    old, new : array_like
        The two domains that determine the map.  Each must (successfully)
        convert to 1-d arrays containing precisely two values.

    Returns
    -------
    x_out : ndarray
        Array of points of the same shape as `x`, after application of the
        linear map between the two domains.

    See Also
    --------
    getdomain, mapparms

    Notes
    -----
    Effectively, this implements:

    .. math::
        x\\_out = new[0] + m(x - old[0])

    where

    .. math::
        m = \\frac{new[1]-new[0]}{old[1]-old[0]}

    Examples
    --------
    >>> from beignet.orthax import polyutils as pu
    >>> old_domain = (-1,1)
    >>> new_domain = (0,2*numpy.pi)
    >>> x = numpy.linspace(-1,1,6); x
    array([-1. , -0.6, -0.2,  0.2,  0.6,  1. ])
    >>> x_out = pu.mapdomain(x, old_domain, new_domain); x_out
    array([ 0.        ,  1.25663706,  2.51327412,  3.76991118,  5.02654825, # may vary
            6.28318531])
    >>> x - pu.mapdomain(x_out, new_domain, old_domain)
    array([0., 0., 0., 0., 0., 0.])

    Also works for complex numbers (and thus can be used to map any line in
    the complex plane to any other line therein).

    >>> i = complex(0,1)
    >>> old = (-1 - i, 1 + i)
    >>> new = (-1 + i, 1 - i)
    >>> z = numpy.linspace(old[0], old[1], 6); z
    array([-1. -1.j , -0.6-0.6j, -0.2-0.2j,  0.2+0.2j,  0.6+0.6j,  1. +1.j ])
    >>> new_z = pu.mapdomain(z, old, new); new_z
    array([-1.0+1.j , -0.6+0.6j, -0.2+0.2j,  0.2-0.2j,  0.6-0.6j,  1.0-1.j ]) # may vary

    """
    x = jax.numpy.asarray(x)
    off, scl = mapparms(old, new)
    return off + scl * x


def _nth_slice(i, ndim):
    sl = [jax.numpy.newaxis] * ndim
    sl[i] = slice(None)
    return tuple(sl)


def _vander_nd(vander_fs, points, degrees):
    r"""A generalization of the Vandermonde matrix for N dimensions.

    The result is built by combining the results of 1d Vandermonde matrices,

    .. math::
        W[i_0, .., i_M, j_0, .., j_N] = \prod_{k=0}^N{V_k(x_k)[i_0, .., i_M, j_k]}

    where

    .. math::
        N &= \texttt{len(points)} = \texttt{len(degrees)} = \texttt{len(vander\_fs)} \\
        M &= \texttt{points[k].ndim} \\
        V_k &= \texttt{vander\_fs[k]} \\
        x_k &= \texttt{points[k]} \\
        0 \le j_k &\le \texttt{degrees[k]}

    Expanding the one-dimensional :math:`V_k` functions gives:

    .. math::
        W[i_0, .., i_M, j_0, .., j_N] = \prod_{k=0}^N{B_{k, j_k}(x_k[i_0, .., i_M])}

    where :math:`B_{k,m}` is the m'th basis of the polynomial construction used along
    dimension :math:`k`. For a regular polynomial, :math:`B_{k, m}(x) = P_m(x) = x^m`.

    Parameters
    ----------
    vander_fs : Sequence[function(array_like, int) -> ndarray]
        The 1d vander function to use for each axis, such as ``polyvander``
    points : Sequence[array_like]
        Arrays of point coordinates, all of the same shape. The dtypes
        will be converted to either float64 or complex128 depending on
        whether any of the elements are complex. Scalars are converted to
        1-D arrays.
        This must be the same length as `vander_fs`.
    degrees : Sequence[int]
        The maximum degree (inclusive) to use for each axis.
        This must be the same length as `vander_fs`.

    Returns
    -------
    vander_nd : ndarray
        An array of shape ``points[0].shape + tuple(d + 1 for d in degrees)``.
    """
    n_dims = len(vander_fs)
    if n_dims != len(points):
        raise ValueError(
            f"Expected {n_dims} dimensions of sample points, got {len(points)}"
        )
    if n_dims != len(degrees):
        raise ValueError(f"Expected {n_dims} dimensions of degrees, got {len(degrees)}")
    if n_dims == 0:
        raise ValueError("Unable to guess a dtype or shape when no points are given")

    # convert to the same shape and type
    points = tuple(jax.numpy.array(tuple(points), copy=False) + 0.0)

    # produce the vandermonde matrix for each dimension, placing the last
    # axis of each in an independent trailing axis of the output
    vander_arrays = (
        vander_fs[i](points[i], degrees[i])[(...,) + _nth_slice(i, n_dims)]
        for i in range(n_dims)
    )

    # we checked this wasn't empty already, so no `initial` needed
    return functools.reduce(operator.mul, vander_arrays)


def _vander_nd_flat(vander_fs, points, degrees):
    """
    Like `_vander_nd`, but flattens the last ``len(degrees)`` axes into a single axis.

    Used to implement the public ``<type>vander<n>d`` functions.
    """
    v = _vander_nd(vander_fs, points, degrees)
    return v.reshape(v.shape[: -len(degrees)] + (-1,))


def _fromroots(line_f, mul_f, roots):
    """
    Helper function used to implement the ``<type>fromroots`` functions.

    Parameters
    ----------
    line_f : function(float, float) -> ndarray
        The ``<type>line`` function, such as ``polyline``
    mul_f : function(array_like, array_like) -> ndarray
        The ``<type>mul`` function, such as ``polymul``
    roots
        See the ``<type>fromroots`` functions for more detail
    """
    roots = jax.numpy.asarray(roots)
    if roots.size == 0:
        return jax.numpy.ones(1)

    roots = jax.numpy.sort(roots)

    retlen = len(roots) + 1

    def p_scan_fun(carry, x):
        return carry, _add(jax.numpy.zeros(retlen, dtype=x.dtype), line_f(-x, 1))

    _, p = jax.lax.scan(p_scan_fun, 0, roots)

    p = jax.numpy.asarray(p)
    n = len(p)

    def cond_fun(val):
        return val[0] > 1

    def body_fun(val):
        m, r = divmod(val[0], 2)
        arr = val[1]
        tmp = jax.numpy.array([jax.numpy.zeros(retlen, dtype=p.dtype)] * len(p))

        def inner_body_fun(i, val):
            return val.at[i].set(mul_f(arr[i], arr[i + m])[:retlen])

        tmp = jax.lax.fori_loop(0, m, inner_body_fun, tmp)
        tmp = jax.lax.cond(
            r, lambda x: x.at[0].set(mul_f(x[0], arr[2 * m])[:retlen]), lambda x: x, tmp
        )

        return (m, tmp)

    _, ret = jax.lax.while_loop(cond_fun, body_fun, (n, p))
    return ret[0]


def _valnd(val_f, c, *args):
    """
    Helper function used to implement the ``<type>val<n>d`` functions.

    Parameters
    ----------
    val_f : function(array_like, array_like, tensor: bool) -> array_like
        The ``<type>val`` function, such as ``polyval``
    c, args
        See the ``<type>val<n>d`` functions for more detail
    """
    args = [jax.numpy.asarray(a) for a in args]
    shape0 = args[0].shape
    if not all(a.shape == shape0 for a in args[1:]):
        if len(args) == 3:
            raise ValueError("x, y, z are incompatible")
        elif len(args) == 2:
            raise ValueError("x, y are incompatible")
        else:
            raise ValueError("ordinates are incompatible")
    it = iter(args)
    x0 = next(it)

    # use tensor on only the first
    c = val_f(x0, c)
    for xi in it:
        c = val_f(xi, c, tensor=False)
    return c


def _gridnd(val_f, c, *args):
    """
    Helper function used to implement the ``<type>grid<n>d`` functions.

    Parameters
    ----------
    val_f : function(array_like, array_like, tensor: bool) -> array_like
        The ``<type>val`` function, such as ``polyval``
    c, args
        See the ``<type>grid<n>d`` functions for more detail
    """
    for xi in args:
        c = val_f(xi, c)
    return c


def _div(mul_f, c1, c2):
    """
    Helper function used to implement the ``<type>div`` functions.

    Implementation uses repeated subtraction of c2 multiplied by the nth basis.
    For some polynomial types, a more efficient approach may be possible.

    Parameters
    ----------
    mul_f : function(array_like, array_like) -> array_like
        The ``<type>mul`` function, such as ``polymul``
    c1, c2
        See the ``<type>div`` functions for more detail
    """
    c1, c2 = as_series(c1, c2)
    lc1 = len(c1)
    lc2 = len(c2)
    if lc1 < lc2:
        return jax.numpy.zeros_like(c1[:1]), c1
    elif lc2 == 1:
        return c1 / c2[-1], jax.numpy.zeros_like(c1[:1])
    else:

        def _ldordidx(x):  # index of highest order nonzero term
            return len(x) - 1 - jax.numpy.nonzero(x[::-1], size=1)[0][0]

        quo = jax.numpy.zeros(lc1 - lc2 + 1, dtype=c1.dtype)
        rem = c1
        ridx = len(rem) - 1
        sz = lc1 - _ldordidx(c2) - 1
        y = jax.numpy.zeros(lc1 + lc2 + 1, dtype=c1.dtype).at[sz].set(1.0)

        def body(k, val):
            quo, rem, y, ridx = val
            i = sz - k
            p = mul_f(y, c2)
            pidx = _ldordidx(p)
            t = rem[ridx] / p[pidx]
            rem = _sub(rem.at[ridx].set(0), t * p.at[pidx].set(0))[: len(rem)]
            quo = quo.at[i].set(t)
            ridx -= 1
            y = jax.numpy.roll(y, -1)
            return quo, rem, y, ridx

        quo, rem, _, _ = jax.lax.fori_loop(0, sz, body, (quo, rem, y, ridx))
        return quo, rem


def _add(c1, c2):
    """Helper function used to implement the ``<type>add`` functions."""
    c1, c2 = as_series(c1, c2)
    if len(c1) > len(c2):
        ret = c1.at[: c2.size].add(c2)
    else:
        ret = c2.at[: c1.size].add(c1)
    return ret


def _sub(c1, c2):
    """Helper function used to implement the ``<type>sub`` functions."""
    c1, c2 = as_series(c1, c2)
    if len(c1) > len(c2):
        ret = c1.at[: c2.size].add(-c2)
    else:
        ret = (-c2).at[: c1.size].add(c1)
    return ret


def _fit(vander_f, x, y, deg, rcond=None, full=False, w=None):  # noqa:C901
    """
    Helper function used to implement the ``<type>fit`` functions.

    Parameters
    ----------
    vander_f : function(array_like, int) -> ndarray
        The 1d vander function, such as ``polyvander``
    c1, c2
        See the ``<type>fit`` functions for more detail
    """
    x = jax.numpy.asarray(x)
    y = jax.numpy.asarray(y)
    deg = numpy.asarray(deg)

    # check arguments.
    if deg.ndim > 1 or deg.dtype.kind not in "iu" or deg.size == 0:
        raise TypeError("deg must be an int or non-empty 1-D array of int")
    if deg.min() < 0:
        raise ValueError("expected deg >= 0")
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")
    if y.ndim < 1 or y.ndim > 2:
        raise TypeError("expected 1D or 2D array for y")
    if len(x) != len(y):
        raise TypeError("expected x and y to have same length")

    if deg.ndim == 0:
        lmax = int(deg)
        van = vander_f(x, lmax)
    else:
        deg = numpy.sort(deg)
        lmax = int(deg[-1])
        van = vander_f(x, lmax)[:, deg]

    # set up the least squares matrices in transposed form
    lhs = van.T
    rhs = y.T
    if w is not None:
        w = jax.numpy.asarray(w)
        if w.ndim != 1:
            raise TypeError("expected 1D vector for w")
        if len(x) != len(w):
            raise TypeError("expected x and w to have same length")
        # apply weights. Don't use inplace operations as they
        # can cause problems with NA.
        lhs = lhs * w
        rhs = rhs * w

    # set rcond
    if rcond is None:
        rcond = len(x) * jax.numpy.finfo(x.dtype).eps

    # Determine the norms of the design matrix columns.
    if issubclass(lhs.dtype.type, jax.numpy.complexfloating):
        scl = jax.numpy.sqrt(
            (jax.numpy.square(lhs.real) + jax.numpy.square(lhs.imag)).sum(1)
        )
    else:
        scl = jax.numpy.sqrt(jax.numpy.square(lhs).sum(1))
    scl = jax.numpy.where(scl == 0, 1, scl)

    # Solve the least squares problem.
    c, resids, rank, s = jax.numpy.linalg.lstsq(lhs.T / scl, rhs.T, rcond)
    c = (c.T / scl).T

    # Expand c to include non-fitted coefficients which are set to zero
    if deg.ndim > 0:
        if c.ndim == 2:
            cc = jax.numpy.zeros((lmax + 1, c.shape[1]), dtype=c.dtype)
        else:
            cc = jax.numpy.zeros(lmax + 1, dtype=c.dtype)
        cc = cc.at[deg].set(c)
        c = cc

    if full:
        return c, [resids, rank, s, rcond]
    else:
        return c


def _pow(mul_f, c, pow, maxpower):
    """
    Helper function used to implement the ``<type>pow`` functions.

    Parameters
    ----------
    mul_f : function(array_like, array_like) -> ndarray
        The ``<type>mul`` function, such as ``polymul``
    c : array_like
        1-D array of array of series coefficients
    pow, maxpower
        See the ``<type>pow`` functions for more detail
    """
    c = as_series(c)
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
        prd = _add(prd, c)

        # This can be made more efficient by using powers of two
        # in the usual way.
        def body(i, p):
            p = mul_f(p, c, mode="same")
            return p

        prd = jax.lax.fori_loop(2, power + 1, body, prd)
        return prd


def _pad_along_axis(array, pad=(0, 0), axis=0):
    """Pad with zeros or truncate a given dimension."""
    array = jax.numpy.moveaxis(array, axis, 0)

    if pad[0] < 0:
        array = array[abs(pad[0]) :]
        pad = (0, pad[1])
    if pad[1] < 0:
        array = array[: -abs(pad[1])]
        pad = (pad[0], 0)

    npad = [(0, 0)] * array.ndim
    npad[0] = pad

    array = jax.numpy.pad(array, pad_width=npad, mode="constant", constant_values=0)
    return jax.numpy.moveaxis(array, 0, axis)
