import functools
import operator
import warnings

import numpy
import numpy.linalg


def _common_type(*xs):
    dtypes = [
        [
            numpy.float16,
            numpy.float32,
            numpy.float64,
        ],
        [
            None,
            numpy.complex64,
            numpy.complex128,
        ],
    ]

    precisions = {
        numpy.float16: 0,
        numpy.float32: 1,
        numpy.float64: 2,
        numpy.complex64: 1,
        numpy.complex128: 2,
    }

    is_complex = False

    precision = 0

    for x in xs:
        if numpy.iscomplexobj(x):
            is_complex = True

        if issubclass(x.dtype.type, numpy.integer):
            score = precisions[numpy.float64]
        else:
            score = precisions.get(x.dtype.type, None)

            if score is None:
                raise TypeError

        precision = max(precision, score)

    if is_complex:
        return dtypes[1][precision]
    else:
        return dtypes[0][precision]


def normalize_axis_index(axis, ndim):
    if axis < 0:
        axis = axis + ndim

    return axis


class RankWarning(RuntimeWarning):
    pass


def _trim_sequence(x):
    if len(x) == 0 or x[-1] != 0:
        output = x
    else:
        for index in range(len(x) - 1, -1, -1):
            if x[index] != 0:
                break

        output = x[: index + 1]

    return output


def _as_series(inputs, trim=True):
    outputs = []

    for input in inputs:
        outputs = [*outputs, numpy.array(input, ndmin=1)]

    for index, output in enumerate(outputs):
        if output.ndim != 1:
            raise ValueError

        if output.size == 0:
            raise ValueError

        if trim:
            output = _trim_sequence(output)

        outputs[index] = output

    try:
        dtype = _common_type(*outputs)
    except Exception as error:
        raise ValueError from error

    for index, output in enumerate(outputs):
        outputs[index] = numpy.array(output, dtype=dtype)

    return outputs


def _trim_coefficients(input, tolerance: float = 0.0):
    if tolerance < 0:
        raise ValueError

    (input,) = _as_series([input])

    [indices] = numpy.nonzero(numpy.abs(input) > tolerance)

    if len(indices) == 0:
        return input[:1] * 0.0
    else:
        return input[: indices[-1] + 1]


chebtrim = _trim_coefficients
hermetrim = _trim_coefficients
hermtrim = _trim_coefficients
lagtrim = _trim_coefficients
legtrim = _trim_coefficients
polytrim = _trim_coefficients


chebdomain = numpy.array([-1.0, 1.0])

chebone = numpy.array([1])

chebx = numpy.array([0, 1])

chebzero = numpy.array([0])


def _add(input, other):
    [input, other] = _as_series([input, other])

    if len(input) > len(other):
        input[: other.size] = input[: other.size] + other

        output = input
    else:
        other[: input.size] = other[: input.size] + input

        output = other

    if len(output) != 0 and output[-1] == 0:
        for index in range(len(output) - 1, -1, -1):
            if output[index] != 0:
                break

        output = output[: index + 1]

    return output


def _c_series_to_z_series(input):
    n = input.size
    output = numpy.zeros(2 * n - 1, dtype=input.dtype)
    output[n - 1 :] = input / 2
    return output + output[::-1]


def _div(func, a, b):
    (
        a,
        b,
    ) = _as_series([a, b])

    if b[-1] == 0:
        raise ZeroDivisionError

    m = a.shape[-1]
    n = b.shape[-1]

    if m < n:
        quotient, remainder = a[:1] * 0.0, a
    elif n == 1:
        quotient, remainder = a / b[-1], a[:1] * 0.0
    else:
        quotient = numpy.empty(m - n + 1, dtype=a.dtype)

        remainder = a

        for index in range(m - n, -1, -1):
            shape = [0] * index

            p = func([*shape, 1], b)

            q = remainder[-1] / p[-1]

            remainder = remainder[:-1] - q * p[:-1]

            quotient[index] = q

        remainder = _trim_sequence(remainder)

    return quotient, remainder


def _evaluate(func, input, *xs):
    xs = [numpy.asanyarray(a) for a in xs]

    if not all((a.shape == xs[0].shape for a in xs[1:])):
        match len(xs):
            case 2:
                raise ValueError("x, y are incompatible")
            case 3:
                raise ValueError("x, y, z are incompatible")
            case _:
                raise ValueError("ordinates are incompatible")

    xs = iter(xs)

    output = func(next(xs), input)

    for x in xs:
        output = func(x, output, tensor=False)

    return output


def _fit(func, x, y, degree, relative_condition=None, full=False, weight=None):
    x = numpy.asarray(x) + 0.0
    y = numpy.asarray(y) + 0.0
    degree = numpy.asarray(degree)

    if degree.ndim > 1 or degree.dtype.kind not in "iu" or degree.size == 0:
        raise TypeError("deg must be an int or non-empty 1-D array of int")
    if degree.min() < 0:
        raise ValueError("expected deg >= 0")
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")
    if y.ndim < 1 or y.ndim > 2:
        raise TypeError("expected 1D or 2D array for y")
    if len(x) != len(y):
        raise TypeError("expected x and y to have same length")

    if degree.ndim == 0:
        lmax = degree
        order = lmax + 1
        van = func(x, lmax)
    else:
        degree = numpy.sort(degree)
        lmax = degree[-1]
        order = len(degree)
        van = func(x, lmax)[:, degree]

    lhs = van.T
    rhs = y.T
    if weight is not None:
        weight = numpy.asarray(weight) + 0.0
        if weight.ndim != 1:
            raise TypeError("expected 1D vector for w")
        if len(x) != len(weight):
            raise TypeError("expected x and w to have same length")

        lhs = lhs * weight
        rhs = rhs * weight

    if relative_condition is None:
        relative_condition = len(x) * numpy.finfo(x.dtype).eps

    if issubclass(lhs.dtype.type, numpy.complexfloating):
        scl = numpy.sqrt((numpy.square(lhs.real) + numpy.square(lhs.imag)).sum(1))
    else:
        scl = numpy.sqrt(numpy.square(lhs).sum(1))
    scl[scl == 0] = 1

    c, resids, rank, s = numpy.linalg.lstsq(lhs.T / scl, rhs.T, relative_condition)
    c = (c.T / scl).T

    if degree.ndim > 0:
        if c.ndim == 2:
            cc = numpy.zeros((lmax + 1, c.shape[1]), dtype=c.dtype)
        else:
            cc = numpy.zeros(lmax + 1, dtype=c.dtype)
        cc[degree] = c
        c = cc

    if rank != order and not full:
        msg = "The fit may be poorly conditioned"
        warnings.warn(msg, RankWarning, stacklevel=2)

    if full:
        return c, [resids, rank, s, relative_condition]
    else:
        return c


def _from_roots(line_f: callable, mul_f: callable, roots):
    if len(roots) == 0:
        return numpy.ones(1)
    else:
        [roots] = _as_series([roots], trim=False)

        roots.sort()

        p = [line_f(-r, 1) for r in roots]

        n = len(p)

        while n > 1:
            m, r = divmod(n, 2)

            tmp = [mul_f(p[i], p[i + m]) for i in range(m)]

            if r:
                tmp[0] = mul_f(tmp[0], p[-1])

            p = tmp

            n = m

        return p[0]


def _grid(func, input, *xs):
    for x in xs:
        input = func(x, input)

    return input


def _normed_hermite_e_n(x, n):
    if n == 0:
        return numpy.full(x.shape, 1 / numpy.sqrt(numpy.sqrt(2 * numpy.pi)))

    c0 = 0.0
    c1 = 1.0 / numpy.sqrt(numpy.sqrt(2 * numpy.pi))
    nd = float(n)
    for _ in range(n - 1):
        tmp = c0
        c0 = -c1 * numpy.sqrt((nd - 1.0) / nd)
        c1 = tmp + c1 * x * numpy.sqrt(1.0 / nd)
        nd = nd - 1.0
    return c0 + c1 * x


def _normed_hermite_n(x, n):
    if n == 0:
        return numpy.full(x.shape, 1 / numpy.sqrt(numpy.sqrt(numpy.pi)))

    c0 = 0.0
    c1 = 1.0 / numpy.sqrt(numpy.sqrt(numpy.pi))
    nd = float(n)
    for _ in range(n - 1):
        tmp = c0
        c0 = -c1 * numpy.sqrt((nd - 1.0) / nd)
        c1 = tmp + c1 * x * numpy.sqrt(2.0 / nd)
        nd = nd - 1.0
    return c0 + c1 * x * numpy.sqrt(2)


def _pow(func, input, exponent, maximum_exponent):
    [input] = _as_series([input])

    exponent = int(exponent)

    if exponent != exponent or exponent < 0:
        raise ValueError("Power must be a non-negative integer.")
    elif maximum_exponent is not None and exponent > maximum_exponent:
        raise ValueError("Power is too large")
    elif exponent == 0:
        return numpy.array([1], dtype=input.dtype)
    elif exponent == 1:
        return input
    else:
        output = input

        for _ in range(2, exponent + 1):
            output = func(output, input)

        return output


def _sub(input, other):
    [input, other] = _as_series([input, other])

    if len(input) > len(other):
        input[: other.size] = input[: other.size] - other

        output = input
    else:
        other = -other

        other[: input.size] = other[: input.size] + input

        output = other

    if len(output) != 0 and output[-1] == 0:
        for index in range(len(output) - 1, -1, -1):
            if output[index] != 0:
                break

        output = output[: index + 1]

    return output


def _vander_nd(func, input, degrees):
    n_dims = len(func)
    if n_dims != len(input):
        raise ValueError(
            f"Expected {n_dims} dimensions of sample points, got {len(input)}"
        )
    if n_dims != len(degrees):
        raise ValueError(f"Expected {n_dims} dimensions of degrees, got {len(degrees)}")
    if n_dims == 0:
        raise ValueError("Unable to guess a dtype or shape when no points are given")

    input = tuple(numpy.asarray(tuple(input)) + 0.0)

    ys = []

    for index in range(n_dims):
        output = [None] * n_dims

        output[index] = slice(None)

        y = func[index](input[index], degrees[index])[(...,) + (*output,)]

        ys = [*ys, y]

    return functools.reduce(operator.mul, ys)


def _vander_nd_flat(vander_fs, points, degrees):
    v = _vander_nd(vander_fs, points, degrees)
    return v.reshape(v.shape[: -len(degrees)] + (-1,))


def _z_series_to_c_series(input):
    n = (input.size + 1) // 2
    output = input[n - 1 :]
    output[1:n] = output[1:n] * 2
    return output


def _zseries_div(z1, z2):
    lc1 = len(z1)
    lc2 = len(z2)
    if lc2 == 1:
        z1 /= z2
        return z1, z1[:1] * 0
    elif lc1 < lc2:
        return z1[:1] * 0, z1
    else:
        dlen = lc1 - lc2
        scl = z2[0]
        z2 /= scl
        quo = numpy.empty(dlen + 1, dtype=z1.dtype)
        i = 0
        j = dlen
        while i < j:
            r = z1[i]
            quo[i] = z1[i]
            quo[dlen - i] = r
            tmp = r * z2
            z1[i : i + lc2] -= tmp
            z1[j : j + lc2] -= tmp
            i += 1
            j -= 1
        r = z1[i]
        quo[i] = r
        tmp = r * z2
        z1[i : i + lc2] -= tmp
        quo /= scl
        rem = z1[i + 1 : i - 1 + lc2]
        return quo, rem


def cheb2poly(input):
    [input] = _as_series([input])

    n = len(input)
    if n < 3:
        return input
    else:
        c0 = input[-2]
        c1 = input[-1]

        for i in range(n - 1, 1, -1):
            tmp = c0
            c0 = polysub(input[i - 2], c1)
            c1 = polyadd(tmp, polymulx(c1) * 2)
        return polyadd(c0, polymulx(c1))


def chebadd(input, other):
    return _add(input, other)


def chebcompanion(c):
    [c] = _as_series([c])
    if len(c) < 2:
        raise ValueError("Series must have maximum degree of at least 1.")
    if len(c) == 2:
        return numpy.array([[-c[0] / c[1]]])

    n = len(c) - 1
    mat = numpy.zeros((n, n), dtype=c.dtype)
    scl = numpy.array([1.0] + [numpy.sqrt(0.5)] * (n - 1))
    top = mat.reshape(-1)[1 :: n + 1]
    bot = mat.reshape(-1)[n :: n + 1]
    top[0] = numpy.sqrt(0.5)
    top[1:] = 1 / 2
    bot[...] = top
    mat[:, -1] -= (c[:-1] / c[-1]) * (scl / scl[-1]) * 0.5
    return mat


def chebder(c, m=1, scl=1, axis=0):
    c = numpy.array(c, ndmin=1)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)
    cnt = operator.index(m)
    iaxis = operator.index(axis)
    if cnt < 0:
        raise ValueError("The order of derivation must be non-negative")
    iaxis = normalize_axis_index(iaxis, c.ndim)

    if cnt == 0:
        return c

    c = numpy.moveaxis(c, iaxis, 0)
    n = len(c)
    if cnt >= n:
        c = c[:1] * 0
    else:
        for _ in range(cnt):
            n = n - 1
            c *= scl
            der = numpy.empty((n,) + c.shape[1:], dtype=c.dtype)
            for j in range(n, 2, -1):
                der[j - 1] = (2 * j) * c[j]
                c[j - 2] += (j * c[j]) / (j - 2)
            if n > 1:
                der[1] = 4 * c[2]
            der[0] = c[1]
            c = der
    c = numpy.moveaxis(c, 0, iaxis)
    return c


def chebdiv(c1, c2):
    [c1, c2] = _as_series([c1, c2])
    if c2[-1] == 0:
        raise ZeroDivisionError()

    lc1 = len(c1)
    lc2 = len(c2)

    if lc1 < lc2:
        return c1[:1] * 0, c1
    elif lc2 == 1:
        return c1 / c2[-1], c1[:1] * 0
    else:
        z1 = _c_series_to_z_series(c1)
        z2 = _c_series_to_z_series(c2)

        quo, rem = _zseries_div(z1, z2)

        quo = _trim_sequence(_z_series_to_c_series(quo))
        rem = _trim_sequence(_z_series_to_c_series(rem))

        return quo, rem


def chebfit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(chebvander, x, y, deg, rcond, full, w)


def chebfromroots(input):
    return _from_roots(chebline, chebmul, input)


def chebgauss(input):
    ideg = operator.index(input)

    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    output = numpy.arange(1, 2 * ideg, 2) / (2.0 * ideg)

    output = output * numpy.pi

    output = numpy.cos(output)

    weight = numpy.ones(ideg) * (numpy.pi / ideg)

    return output, weight


def chebgrid2d(x, y, c):
    return _grid(chebval, c, x, y)


def chebgrid3d(x, y, z, c):
    return _grid(chebval, c, x, y, z)


def chebint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    if k is None:
        k = []
    c = numpy.array(c, ndmin=1)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)
    if not numpy.iterable(k):
        k = [k]
    cnt = operator.index(m)
    iaxis = operator.index(axis)
    if cnt < 0:
        raise ValueError("The order of integration must be non-negative")
    if len(k) > cnt:
        raise ValueError("Too many integration constants")
    if numpy.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if numpy.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")
    iaxis = normalize_axis_index(iaxis, c.ndim)

    if cnt == 0:
        return c

    c = numpy.moveaxis(c, iaxis, 0)
    k = list(k) + [0] * (cnt - len(k))
    for i in range(cnt):
        n = len(c)
        c *= scl
        if n == 1 and numpy.all(c[0] == 0):
            c[0] += k[i]
        else:
            tmp = numpy.empty((n + 1,) + c.shape[1:], dtype=c.dtype)
            tmp[0] = c[0] * 0
            tmp[1] = c[0]
            if n > 1:
                tmp[2] = c[1] / 4
            for j in range(2, n):
                tmp[j + 1] = c[j] / (2 * (j + 1))
                tmp[j - 1] -= c[j] / (2 * (j - 1))
            tmp[0] += k[i] - chebval(lbnd, tmp)
            c = tmp
    c = numpy.moveaxis(c, 0, iaxis)
    return c


def chebinterpolate(func, deg, args=()):
    deg = numpy.asarray(deg)

    if deg.ndim > 0 or deg.dtype.kind not in "iu" or deg.size == 0:
        raise TypeError("deg must be an int")
    if deg < 0:
        raise ValueError("expected deg >= 0")

    order = deg + 1
    xcheb = chebpts1(order)
    yfunc = func(xcheb, *args)
    m = chebvander(xcheb, deg)
    c = numpy.dot(m.T, yfunc)
    c[0] /= order
    c[1:] /= 0.5 * order

    return c


def chebline(off, scl):
    if scl != 0:
        return numpy.array([off, scl])
    else:
        return numpy.array([off])


def chebmul(input, other):
    [input, other] = _as_series([input, other])

    input = _c_series_to_z_series(input)
    other = _c_series_to_z_series(other)

    output = numpy.convolve(input, other)

    n = (output.size + 1) // 2

    output = output[n - 1 :]

    output[1:n] = output[1:n] * 2

    if len(output) != 0 and output[-1] == 0:
        for index in range(len(output) - 1, -1, -1):
            if output[index] != 0:
                break

        output = output[: index + 1]

    return output


def chebmulx(c):
    [c] = _as_series([c])

    if len(c) == 1 and c[0] == 0:
        return c

    prd = numpy.empty(len(c) + 1, dtype=c.dtype)
    prd[0] = c[0] * 0
    prd[1] = c[0]
    if len(c) > 1:
        tmp = c[1:] / 2
        prd[2:] = tmp
        prd[0:-2] += tmp
    return prd


def chebpow(c, pow, maxpower=16):
    [c] = _as_series([c])
    power = int(pow)
    if power != pow or power < 0:
        raise ValueError("Power must be a non-negative integer.")
    elif maxpower is not None and power > maxpower:
        raise ValueError("Power is too large")
    elif power == 0:
        return numpy.array([1], dtype=c.dtype)
    elif power == 1:
        return c
    else:
        zs = _c_series_to_z_series(c)
        prd = zs
        for _ in range(2, power + 1):
            prd = numpy.convolve(prd, zs)
        return _z_series_to_c_series(prd)


def chebpts1(npts):
    _npts = int(npts)
    if _npts != npts:
        raise ValueError("npts must be integer")
    if _npts < 1:
        raise ValueError("npts must be >= 1")

    x = 0.5 * numpy.pi / _npts * numpy.arange(-_npts + 1, _npts + 1, 2)
    return numpy.sin(x)


def chebpts2(npts):
    _npts = int(npts)
    if _npts != npts:
        raise ValueError("npts must be integer")
    if _npts < 2:
        raise ValueError("npts must be >= 2")

    x = numpy.linspace(-numpy.pi, 0, _npts)
    return numpy.cos(x)


def chebroots(c):
    [c] = _as_series([c])
    if len(c) < 2:
        return numpy.array([], dtype=c.dtype)
    if len(c) == 2:
        return numpy.array([-c[0] / c[1]])

    m = chebcompanion(c)[::-1, ::-1]
    r = numpy.linalg.eigvals(m)
    r.sort()
    return r


def chebsub(c1, c2):
    return _sub(c1, c2)


def chebval(x, c, tensor=True):
    c = numpy.array(c, ndmin=1)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)
    if isinstance(x, (tuple, list)):
        x = numpy.asarray(x)
    if isinstance(x, numpy.ndarray) and tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)

    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        x2 = 2 * x
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            c0 = c[-i] - c1
            c1 = tmp + c1 * x2
    return c0 + c1 * x


def chebval2d(x, y, c):
    return _evaluate(chebval, c, x, y)


def chebval3d(x, y, z, c):
    return _evaluate(chebval, c, x, y, z)


def chebvander(x, deg):
    ideg = operator.index(deg)
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    x = numpy.array(x, ndmin=1) + 0.0
    dims = (ideg + 1,) + x.shape
    dtyp = x.dtype
    v = numpy.empty(dims, dtype=dtyp)

    v[0] = x * 0 + 1
    if ideg > 0:
        x2 = 2 * x
        v[1] = x
        for i in range(2, ideg + 1):
            v[i] = v[i - 1] * x2 - v[i - 2]
    return numpy.moveaxis(v, 0, -1)


def chebvander2d(x, y, deg):
    return _vander_nd_flat((chebvander, chebvander), (x, y), deg)


def chebvander3d(x, y, z, deg):
    return _vander_nd_flat((chebvander, chebvander, chebvander), (x, y, z), deg)


def chebweight(x):
    w = 1.0 / (numpy.sqrt(1.0 + x) * numpy.sqrt(1.0 - x))
    return w


def _get_domain(x):
    [x] = _as_series([x], trim=False)

    if x.dtype.char in numpy.typecodes["Complex"]:
        rmin, rmax = x.real.min(), x.real.max()
        imin, imax = x.imag.min(), x.imag.max()
        return numpy.array((complex(rmin, imin), complex(rmax, imax)))
    else:
        return numpy.array((x.min(), x.max()))


def herm2poly(input):
    [input] = _as_series([input])
    n = len(input)
    if n == 1:
        return input
    if n == 2:
        input[1] *= 2
        return input
    else:
        c0 = input[-2]
        c1 = input[-1]

        for i in range(n - 1, 1, -1):
            tmp = c0
            c0 = polysub(input[i - 2], c1 * (2 * (i - 1)))
            c1 = polyadd(tmp, polymulx(c1) * 2)
        return polyadd(c0, polymulx(c1) * 2)


def hermadd(c1, c2):
    return _add(c1, c2)


def hermcompanion(c):
    [c] = _as_series([c])
    if len(c) < 2:
        raise ValueError("Series must have maximum degree of at least 1.")
    if len(c) == 2:
        return numpy.array([[-0.5 * c[0] / c[1]]])

    n = len(c) - 1
    mat = numpy.zeros((n, n), dtype=c.dtype)
    scl = numpy.hstack((1.0, 1.0 / numpy.sqrt(2.0 * numpy.arange(n - 1, 0, -1))))
    scl = numpy.multiply.accumulate(scl)[::-1]
    top = mat.reshape(-1)[1 :: n + 1]
    bot = mat.reshape(-1)[n :: n + 1]
    top[...] = numpy.sqrt(0.5 * numpy.arange(1, n))
    bot[...] = top
    mat[:, -1] -= scl * c[:-1] / (2.0 * c[-1])
    return mat


def hermder(c, m=1, scl=1, axis=0):
    c = numpy.array(c, ndmin=1)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)
    cnt = operator.index(m)
    iaxis = operator.index(axis)
    if cnt < 0:
        raise ValueError("The order of derivation must be non-negative")
    iaxis = normalize_axis_index(iaxis, c.ndim)

    if cnt == 0:
        return c

    c = numpy.moveaxis(c, iaxis, 0)
    n = len(c)
    if cnt >= n:
        c = c[:1] * 0
    else:
        for _ in range(cnt):
            n = n - 1
            c *= scl
            der = numpy.empty((n,) + c.shape[1:], dtype=c.dtype)
            for j in range(n, 0, -1):
                der[j - 1] = (2 * j) * c[j]
            c = der
    c = numpy.moveaxis(c, 0, iaxis)
    return c


def hermdiv(c1, c2):
    return _div(hermmul, c1, c2)


def herme2poly(input):
    [input] = _as_series([input])
    n = len(input)
    if n == 1:
        return input
    if n == 2:
        return input
    else:
        c0 = input[-2]
        c1 = input[-1]

        for i in range(n - 1, 1, -1):
            tmp = c0
            c0 = polysub(input[i - 2], c1 * (i - 1))
            c1 = polyadd(tmp, polymulx(c1))
        return polyadd(c0, polymulx(c1))


def hermeadd(c1, c2):
    return _add(c1, c2)


def hermecompanion(c):
    [c] = _as_series([c])
    if len(c) < 2:
        raise ValueError("Series must have maximum degree of at least 1.")
    if len(c) == 2:
        return numpy.array([[-c[0] / c[1]]])

    n = len(c) - 1
    mat = numpy.zeros((n, n), dtype=c.dtype)
    scl = numpy.hstack((1.0, 1.0 / numpy.sqrt(numpy.arange(n - 1, 0, -1))))
    scl = numpy.multiply.accumulate(scl)[::-1]
    top = mat.reshape(-1)[1 :: n + 1]
    bot = mat.reshape(-1)[n :: n + 1]
    top[...] = numpy.sqrt(numpy.arange(1, n))
    bot[...] = top
    mat[:, -1] -= scl * c[:-1] / c[-1]
    return mat


def hermeder(c, m=1, scl=1, axis=0):
    c = numpy.array(c, ndmin=1)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)
    cnt = operator.index(m)
    iaxis = operator.index(axis)
    if cnt < 0:
        raise ValueError("The order of derivation must be non-negative")
    iaxis = normalize_axis_index(iaxis, c.ndim)

    if cnt == 0:
        return c

    c = numpy.moveaxis(c, iaxis, 0)
    n = len(c)
    if cnt >= n:
        return c[:1] * 0
    else:
        for _ in range(cnt):
            n = n - 1
            c *= scl
            der = numpy.empty((n,) + c.shape[1:], dtype=c.dtype)
            for j in range(n, 0, -1):
                der[j - 1] = j * c[j]
            c = der
    c = numpy.moveaxis(c, 0, iaxis)
    return c


def hermediv(c1, c2):
    return _div(hermemul, c1, c2)


def hermefit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(hermevander, x, y, deg, rcond, full, w)


def hermefromroots(input):
    return _from_roots(hermeline, hermemul, input)


def hermegauss(input):
    ideg = operator.index(input)
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    c = numpy.array([0] * input + [1])
    m = hermecompanion(c)
    x = numpy.linalg.eigvalsh(m)

    dy = _normed_hermite_e_n(x, ideg)
    df = _normed_hermite_e_n(x, ideg - 1) * numpy.sqrt(ideg)
    x -= dy / df

    fm = _normed_hermite_e_n(x, ideg - 1)
    fm /= numpy.abs(fm).max()
    w = 1 / (fm * fm)

    w = (w + w[::-1]) / 2
    x = (x - x[::-1]) / 2

    w *= numpy.sqrt(2 * numpy.pi) / w.sum()

    return x, w


def hermegrid2d(x, y, c):
    return _grid(hermeval, c, x, y)


def hermegrid3d(x, y, z, c):
    return _grid(hermeval, c, x, y, z)


def hermeint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    if k is None:
        k = []
    c = numpy.array(c, ndmin=1)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)
    if not numpy.iterable(k):
        k = [k]
    cnt = operator.index(m)
    iaxis = operator.index(axis)
    if cnt < 0:
        raise ValueError("The order of integration must be non-negative")
    if len(k) > cnt:
        raise ValueError("Too many integration constants")
    if numpy.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if numpy.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")
    iaxis = normalize_axis_index(iaxis, c.ndim)

    if cnt == 0:
        return c

    c = numpy.moveaxis(c, iaxis, 0)
    k = list(k) + [0] * (cnt - len(k))
    for i in range(cnt):
        n = len(c)
        c *= scl
        if n == 1 and numpy.all(c[0] == 0):
            c[0] += k[i]
        else:
            tmp = numpy.empty((n + 1,) + c.shape[1:], dtype=c.dtype)
            tmp[0] = c[0] * 0
            tmp[1] = c[0]
            for j in range(1, n):
                tmp[j + 1] = c[j] / (j + 1)
            tmp[0] += k[i] - hermeval(lbnd, tmp)
            c = tmp
    c = numpy.moveaxis(c, 0, iaxis)
    return c


def hermeline(off, scl):
    if scl != 0:
        return numpy.array([off, scl])
    else:
        return numpy.array([off])


def hermemul(c1, c2):
    [c1, c2] = _as_series([c1, c2])

    if len(c1) > len(c2):
        c = c2
        xs = c1
    else:
        c = c1
        xs = c2

    if len(c) == 1:
        c0 = c[0] * xs
        c1 = 0
    elif len(c) == 2:
        c0 = c[0] * xs
        c1 = c[1] * xs
    else:
        nd = len(c)
        c0 = c[-2] * xs
        c1 = c[-1] * xs
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = hermesub(c[-i] * xs, c1 * (nd - 1))
            c1 = hermeadd(tmp, hermemulx(c1))
    return hermeadd(c0, hermemulx(c1))


def hermemulx(c):
    [c] = _as_series([c])

    if len(c) == 1 and c[0] == 0:
        return c

    prd = numpy.empty(len(c) + 1, dtype=c.dtype)
    prd[0] = c[0] * 0
    prd[1] = c[0]
    for i in range(1, len(c)):
        prd[i + 1] = c[i]
        prd[i - 1] += c[i] * i
    return prd


def hermepow(c, pow, maxpower=16):
    return _pow(hermemul, c, pow, maxpower)


def hermeroots(input):
    [input] = _as_series([input])
    if len(input) <= 1:
        return numpy.array([], dtype=input.dtype)
    if len(input) == 2:
        return numpy.array([-input[0] / input[1]])

    m = hermecompanion(input)[::-1, ::-1]
    r = numpy.linalg.eigvals(m)
    r.sort()
    return r


def hermesub(c1, c2):
    return _sub(c1, c2)


def hermeval(x, c, tensor=True):
    c = numpy.array(c, ndmin=1)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)
    if isinstance(x, (tuple, list)):
        x = numpy.asarray(x)
    if isinstance(x, numpy.ndarray) and tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)

    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - c1 * (nd - 1)
            c1 = tmp + c1 * x
    return c0 + c1 * x


def hermeval2d(x, y, c):
    return _evaluate(hermeval, c, x, y)


def hermeval3d(x, y, z, c):
    return _evaluate(hermeval, c, x, y, z)


def hermevander(x, deg):
    ideg = operator.index(deg)
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    x = numpy.array(x, ndmin=1) + 0.0
    dims = (ideg + 1,) + x.shape
    dtyp = x.dtype
    v = numpy.empty(dims, dtype=dtyp)
    v[0] = x * 0 + 1
    if ideg > 0:
        v[1] = x
        for i in range(2, ideg + 1):
            v[i] = v[i - 1] * x - v[i - 2] * (i - 1)
    return numpy.moveaxis(v, 0, -1)


def hermevander2d(x, y, deg):
    return _vander_nd_flat((hermevander, hermevander), (x, y), deg)


def hermevander3d(x, y, z, deg):
    return _vander_nd_flat((hermevander, hermevander, hermevander), (x, y, z), deg)


def hermeweight(x):
    w = numpy.exp(-0.5 * x**2)
    return w


def hermfit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(hermvander, x, y, deg, rcond, full, w)


def hermfromroots(input):
    return _from_roots(hermline, hermmul, input)


def hermgauss(input):
    ideg = operator.index(input)
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    c = numpy.array([0] * input + [1], dtype=numpy.float64)
    m = hermcompanion(c)
    output = numpy.linalg.eigvalsh(m)

    dy = _normed_hermite_n(output, ideg)
    df = _normed_hermite_n(output, ideg - 1) * numpy.sqrt(2 * ideg)
    output -= dy / df

    fm = _normed_hermite_n(output, ideg - 1)
    fm /= numpy.abs(fm).max()
    weight = 1 / (fm * fm)

    weight = (weight + weight[::-1]) / 2
    output = (output - output[::-1]) / 2

    weight = weight * (numpy.sqrt(numpy.pi) / numpy.sum(weight))

    return output, weight


def hermgrid2d(x, y, c):
    return _grid(hermval, c, x, y)


def hermgrid3d(x, y, z, c):
    return _grid(hermval, c, x, y, z)


def hermint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    if k is None:
        k = []
    c = numpy.array(c, ndmin=1)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)
    if not numpy.iterable(k):
        k = [k]
    cnt = operator.index(m)
    iaxis = operator.index(axis)
    if cnt < 0:
        raise ValueError("The order of integration must be non-negative")
    if len(k) > cnt:
        raise ValueError("Too many integration constants")
    if numpy.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if numpy.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")
    iaxis = normalize_axis_index(iaxis, c.ndim)

    if cnt == 0:
        return c

    c = numpy.moveaxis(c, iaxis, 0)
    k = list(k) + [0] * (cnt - len(k))
    for i in range(cnt):
        n = len(c)
        c *= scl
        if n == 1 and numpy.all(c[0] == 0):
            c[0] += k[i]
        else:
            tmp = numpy.empty((n + 1,) + c.shape[1:], dtype=c.dtype)
            tmp[0] = c[0] * 0
            tmp[1] = c[0] / 2
            for j in range(1, n):
                tmp[j + 1] = c[j] / (2 * (j + 1))
            tmp[0] += k[i] - hermval(lbnd, tmp)
            c = tmp
    c = numpy.moveaxis(c, 0, iaxis)
    return c


def hermline(off, scl):
    if scl != 0:
        return numpy.array([off, scl / 2])
    else:
        return numpy.array([off])


def hermmul(c1, c2):
    [c1, c2] = _as_series([c1, c2])

    if len(c1) > len(c2):
        c = c2
        xs = c1
    else:
        c = c1
        xs = c2

    if len(c) == 1:
        c0 = c[0] * xs
        c1 = 0
    elif len(c) == 2:
        c0 = c[0] * xs
        c1 = c[1] * xs
    else:
        nd = len(c)
        c0 = c[-2] * xs
        c1 = c[-1] * xs
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = hermsub(c[-i] * xs, c1 * (2 * (nd - 1)))
            c1 = hermadd(tmp, hermmulx(c1) * 2)
    return hermadd(c0, hermmulx(c1) * 2)


def hermmulx(c):
    [c] = _as_series([c])

    if len(c) == 1 and c[0] == 0:
        return c

    prd = numpy.empty(len(c) + 1, dtype=c.dtype)
    prd[0] = c[0] * 0
    prd[1] = c[0] / 2
    for i in range(1, len(c)):
        prd[i + 1] = c[i] / 2
        prd[i - 1] += c[i] * i
    return prd


def hermpow(c, pow, maxpower=16):
    return _pow(hermmul, c, pow, maxpower)


def hermroots(input):
    [input] = _as_series([input])
    if len(input) <= 1:
        return numpy.array([], dtype=input.dtype)
    if len(input) == 2:
        return numpy.array([-0.5 * input[0] / input[1]])

    m = hermcompanion(input)[::-1, ::-1]
    r = numpy.linalg.eigvals(m)
    r.sort()
    return r


def hermsub(c1, c2):
    return _sub(c1, c2)


def hermval(x, c, tensor=True):
    c = numpy.array(c, ndmin=1)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)
    if isinstance(x, (tuple, list)):
        x = numpy.asarray(x)
    if isinstance(x, numpy.ndarray) and tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)

    x2 = x * 2
    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - c1 * (2 * (nd - 1))
            c1 = tmp + c1 * x2
    return c0 + c1 * x2


def hermval2d(x, y, c):
    return _evaluate(hermval, c, x, y)


def hermval3d(x, y, z, c):
    return _evaluate(hermval, c, x, y, z)


def hermvander(x, deg):
    ideg = operator.index(deg)
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    x = numpy.array(x, ndmin=1) + 0.0
    dims = (ideg + 1,) + x.shape
    dtyp = x.dtype
    v = numpy.empty(dims, dtype=dtyp)
    v[0] = x * 0 + 1
    if ideg > 0:
        x2 = x * 2
        v[1] = x2
        for i in range(2, ideg + 1):
            v[i] = v[i - 1] * x2 - v[i - 2] * (2 * (i - 1))
    return numpy.moveaxis(v, 0, -1)


def hermvander2d(x, y, deg):
    return _vander_nd_flat((hermvander, hermvander), (x, y), deg)


def hermvander3d(x, y, z, deg):
    return _vander_nd_flat((hermvander, hermvander, hermvander), (x, y, z), deg)


def hermweight(x):
    w = numpy.exp(-(x**2))
    return w


def lag2poly(c):
    [c] = _as_series([c])
    n = len(c)
    if n == 1:
        return c
    else:
        c0 = c[-2]
        c1 = c[-1]

        for i in range(n - 1, 1, -1):
            tmp = c0
            c0 = polysub(c[i - 2], (c1 * (i - 1)) / i)
            c1 = polyadd(tmp, polysub((2 * i - 1) * c1, polymulx(c1)) / i)
        return polyadd(c0, polysub(c1, polymulx(c1)))


def lagadd(c1, c2):
    return _add(c1, c2)


def lagcompanion(c):
    [c] = _as_series([c])
    if len(c) < 2:
        raise ValueError("Series must have maximum degree of at least 1.")
    if len(c) == 2:
        return numpy.array([[1 + c[0] / c[1]]])

    n = len(c) - 1
    mat = numpy.zeros((n, n), dtype=c.dtype)
    top = mat.reshape(-1)[1 :: n + 1]
    mid = mat.reshape(-1)[0 :: n + 1]
    bot = mat.reshape(-1)[n :: n + 1]
    top[...] = -numpy.arange(1, n)
    mid[...] = 2.0 * numpy.arange(n) + 1.0
    bot[...] = top
    mat[:, -1] += (c[:-1] / c[-1]) * n
    return mat


def lagder(c, m=1, scl=1, axis=0):
    c = numpy.array(c, ndmin=1)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)

    cnt = operator.index(m)
    iaxis = operator.index(axis)
    if cnt < 0:
        raise ValueError("The order of derivation must be non-negative")
    iaxis = normalize_axis_index(iaxis, c.ndim)

    if cnt == 0:
        return c

    c = numpy.moveaxis(c, iaxis, 0)
    n = len(c)
    if cnt >= n:
        c = c[:1] * 0
    else:
        for _ in range(cnt):
            n = n - 1
            c *= scl
            der = numpy.empty((n,) + c.shape[1:], dtype=c.dtype)
            for j in range(n, 1, -1):
                der[j - 1] = -c[j]
                c[j - 1] += c[j]
            der[0] = -c[1]
            c = der
    c = numpy.moveaxis(c, 0, iaxis)
    return c


def lagdiv(c1, c2):
    return _div(lagmul, c1, c2)


def lagfit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(lagvander, x, y, deg, rcond, full, w)


def lagfromroots(roots):
    return _from_roots(lagline, lagmul, roots)


def laggauss(input):
    ideg = operator.index(input)
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    c = numpy.array([0] * input + [1])
    m = lagcompanion(c)
    x = numpy.linalg.eigvalsh(m)

    dy = lagval(x, c)
    df = lagval(x, lagder(c))
    x -= dy / df

    fm = lagval(x, c[1:])

    fm = fm / numpy.max(numpy.abs(fm))
    df = df / numpy.max(numpy.abs(df))

    weight = 1.0 / (fm * df)

    weight = weight / numpy.sum(weight)

    return x, weight


def laggrid2d(x, y, c):
    return _grid(lagval, c, x, y)


def laggrid3d(x, y, z, c):
    return _grid(lagval, c, x, y, z)


def lagint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    if k is None:
        k = []
    c = numpy.array(c, ndmin=1)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)
    if not numpy.iterable(k):
        k = [k]
    cnt = operator.index(m)
    iaxis = operator.index(axis)
    if cnt < 0:
        raise ValueError("The order of integration must be non-negative")
    if len(k) > cnt:
        raise ValueError("Too many integration constants")
    if numpy.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if numpy.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")
    iaxis = normalize_axis_index(iaxis, c.ndim)

    if cnt == 0:
        return c

    c = numpy.moveaxis(c, iaxis, 0)
    k = list(k) + [0] * (cnt - len(k))
    for i in range(cnt):
        n = len(c)
        c *= scl
        if n == 1 and numpy.all(c[0] == 0):
            c[0] += k[i]
        else:
            tmp = numpy.empty((n + 1,) + c.shape[1:], dtype=c.dtype)
            tmp[0] = c[0]
            tmp[1] = -c[0]
            for j in range(1, n):
                tmp[j] += c[j]
                tmp[j + 1] = -c[j]
            tmp[0] += k[i] - lagval(lbnd, tmp)
            c = tmp
    c = numpy.moveaxis(c, 0, iaxis)
    return c


def lagline(off, scl):
    if scl != 0:
        return numpy.array([off + scl, -scl])
    else:
        return numpy.array([off])


def lagmul(c1, c2):
    [c1, c2] = _as_series([c1, c2])

    if len(c1) > len(c2):
        c = c2
        xs = c1
    else:
        c = c1
        xs = c2

    if len(c) == 1:
        c0 = c[0] * xs
        c1 = 0
    elif len(c) == 2:
        c0 = c[0] * xs
        c1 = c[1] * xs
    else:
        nd = len(c)
        c0 = c[-2] * xs
        c1 = c[-1] * xs
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = lagsub(c[-i] * xs, (c1 * (nd - 1)) / nd)
            c1 = lagadd(tmp, lagsub((2 * nd - 1) * c1, lagmulx(c1)) / nd)
    return lagadd(c0, lagsub(c1, lagmulx(c1)))


def lagmulx(c):
    [c] = _as_series([c])

    if len(c) == 1 and c[0] == 0:
        return c

    prd = numpy.empty(len(c) + 1, dtype=c.dtype)
    prd[0] = c[0]
    prd[1] = -c[0]
    for i in range(1, len(c)):
        prd[i + 1] = -c[i] * (i + 1)
        prd[i] += c[i] * (2 * i + 1)
        prd[i - 1] -= c[i] * i
    return prd


def lagpow(c, pow, maxpower=16):
    return _pow(lagmul, c, pow, maxpower)


def lagroots(c):
    [c] = _as_series([c])
    if len(c) <= 1:
        return numpy.array([], dtype=c.dtype)
    if len(c) == 2:
        return numpy.array([1 + c[0] / c[1]])

    m = lagcompanion(c)[::-1, ::-1]
    r = numpy.linalg.eigvals(m)
    r.sort()
    return r


def lagsub(c1, c2):
    return _sub(c1, c2)


def lagval(x, c, tensor=True):
    c = numpy.array(c, ndmin=1)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)
    if isinstance(x, (tuple, list)):
        x = numpy.asarray(x)
    if isinstance(x, numpy.ndarray) and tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)

    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - (c1 * (nd - 1)) / nd
            c1 = tmp + (c1 * ((2 * nd - 1) - x)) / nd
    return c0 + c1 * (1 - x)


def lagval2d(x, y, c):
    return _evaluate(lagval, c, x, y)


def lagval3d(x, y, z, c):
    return _evaluate(lagval, c, x, y, z)


def lagvander(x, deg):
    ideg = operator.index(deg)
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    x = numpy.array(x, ndmin=1) + 0.0
    dims = (ideg + 1,) + x.shape
    dtyp = x.dtype
    v = numpy.empty(dims, dtype=dtyp)
    v[0] = x * 0 + 1
    if ideg > 0:
        v[1] = 1 - x
        for i in range(2, ideg + 1):
            v[i] = (v[i - 1] * (2 * i - 1 - x) - v[i - 2] * (i - 1)) / i
    return numpy.moveaxis(v, 0, -1)


def lagvander2d(x, y, deg):
    return _vander_nd_flat((lagvander, lagvander), (x, y), deg)


def lagvander3d(x, y, z, deg):
    return _vander_nd_flat((lagvander, lagvander, lagvander), (x, y, z), deg)


def lagweight(x):
    w = numpy.exp(-x)
    return w


def leg2poly(c):
    [c] = _as_series([c])
    n = len(c)
    if n < 3:
        return c
    else:
        c0 = c[-2]
        c1 = c[-1]

        for i in range(n - 1, 1, -1):
            tmp = c0
            c0 = polysub(c[i - 2], (c1 * (i - 1)) / i)
            c1 = polyadd(tmp, (polymulx(c1) * (2 * i - 1)) / i)
        return polyadd(c0, polymulx(c1))


def legadd(c1, c2):
    return _add(c1, c2)


def legcompanion(c):
    [c] = _as_series([c])
    if len(c) < 2:
        raise ValueError("Series must have maximum degree of at least 1.")
    if len(c) == 2:
        return numpy.array([[-c[0] / c[1]]])

    n = len(c) - 1
    mat = numpy.zeros((n, n), dtype=c.dtype)
    scl = 1.0 / numpy.sqrt(2 * numpy.arange(n) + 1)
    top = mat.reshape(-1)[1 :: n + 1]
    bot = mat.reshape(-1)[n :: n + 1]
    top[...] = numpy.arange(1, n) * scl[: n - 1] * scl[1:n]
    bot[...] = top
    mat[:, -1] -= (c[:-1] / c[-1]) * (scl / scl[-1]) * (n / (2 * n - 1))
    return mat


def legder(c, m=1, scl=1, axis=0):
    c = numpy.array(c, ndmin=1)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)
    cnt = operator.index(m)
    iaxis = operator.index(axis)
    if cnt < 0:
        raise ValueError("The order of derivation must be non-negative")
    iaxis = normalize_axis_index(iaxis, c.ndim)

    if cnt == 0:
        return c

    c = numpy.moveaxis(c, iaxis, 0)
    n = len(c)
    if cnt >= n:
        c = c[:1] * 0
    else:
        for _ in range(cnt):
            n = n - 1
            c *= scl
            der = numpy.empty((n,) + c.shape[1:], dtype=c.dtype)
            for j in range(n, 2, -1):
                der[j - 1] = (2 * j - 1) * c[j]
                c[j - 2] += c[j]
            if n > 1:
                der[1] = 3 * c[2]
            der[0] = c[1]
            c = der
    c = numpy.moveaxis(c, 0, iaxis)
    return c


def legdiv(c1, c2):
    return _div(legmul, c1, c2)


def legfit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(legvander, x, y, deg, rcond, full, w)


def legfromroots(roots):
    return _from_roots(legline, legmul, roots)


def leggauss(input):
    ideg = operator.index(input)
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    c = numpy.array([0] * input + [1])
    m = legcompanion(c)
    x = numpy.linalg.eigvalsh(m)

    dy = legval(x, c)
    df = legval(x, legder(c))
    x -= dy / df

    fm = legval(x, c[1:])
    fm /= numpy.abs(fm).max()
    df /= numpy.abs(df).max()
    weights = 1 / (fm * df)

    weights = (weights + weights[::-1]) / 2
    x = (x - x[::-1]) / 2

    weights = weights * (2.0 / weights.sum())

    return x, weights


def leggrid2d(x, y, c):
    return _grid(legval, c, x, y)


def leggrid3d(x, y, z, c):
    return _grid(legval, c, x, y, z)


def legint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    if k is None:
        k = []
    c = numpy.array(c, ndmin=1)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)
    if not numpy.iterable(k):
        k = [k]
    cnt = operator.index(m)
    iaxis = operator.index(axis)
    if cnt < 0:
        raise ValueError("The order of integration must be non-negative")
    if len(k) > cnt:
        raise ValueError("Too many integration constants")
    if numpy.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if numpy.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")
    iaxis = normalize_axis_index(iaxis, c.ndim)

    if cnt == 0:
        return c

    c = numpy.moveaxis(c, iaxis, 0)
    k = list(k) + [0] * (cnt - len(k))
    for i in range(cnt):
        n = len(c)
        c *= scl
        if n == 1 and numpy.all(c[0] == 0):
            c[0] += k[i]
        else:
            tmp = numpy.empty((n + 1,) + c.shape[1:], dtype=c.dtype)
            tmp[0] = c[0] * 0
            tmp[1] = c[0]
            if n > 1:
                tmp[2] = c[1] / 3
            for j in range(2, n):
                t = c[j] / (2 * j + 1)
                tmp[j + 1] = t
                tmp[j - 1] -= t
            tmp[0] += k[i] - legval(lbnd, tmp)
            c = tmp
    c = numpy.moveaxis(c, 0, iaxis)
    return c


def legline(off, scl):
    if scl != 0:
        return numpy.array([off, scl])
    else:
        return numpy.array([off])


def legmul(c1, c2):
    [c1, c2] = _as_series([c1, c2])

    if len(c1) > len(c2):
        c = c2
        xs = c1
    else:
        c = c1
        xs = c2

    if len(c) == 1:
        c0 = c[0] * xs
        c1 = 0
    elif len(c) == 2:
        c0 = c[0] * xs
        c1 = c[1] * xs
    else:
        nd = len(c)
        c0 = c[-2] * xs
        c1 = c[-1] * xs
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = legsub(c[-i] * xs, (c1 * (nd - 1)) / nd)
            c1 = legadd(tmp, (legmulx(c1) * (2 * nd - 1)) / nd)
    return legadd(c0, legmulx(c1))


def legmulx(c):
    [c] = _as_series([c])

    if len(c) == 1 and c[0] == 0:
        return c

    prd = numpy.empty(len(c) + 1, dtype=c.dtype)
    prd[0] = c[0] * 0
    prd[1] = c[0]
    for i in range(1, len(c)):
        j = i + 1
        k = i - 1
        s = i + j
        prd[j] = (c[i] * j) / s
        prd[k] += (c[i] * i) / s
    return prd


def legpow(c, pow, maxpower=16):
    return _pow(legmul, c, pow, maxpower)


def legroots(c):
    [c] = _as_series([c])
    if len(c) < 2:
        return numpy.array([], dtype=c.dtype)
    if len(c) == 2:
        return numpy.array([-c[0] / c[1]])

    m = legcompanion(c)[::-1, ::-1]
    r = numpy.linalg.eigvals(m)
    r.sort()
    return r


def legsub(c1, c2):
    return _sub(c1, c2)


def legval(x, c, tensor=True):
    c = numpy.array(c, ndmin=1)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c.astype(numpy.double)
    if isinstance(x, (tuple, list)):
        x = numpy.asarray(x)
    if isinstance(x, numpy.ndarray) and tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)

    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - (c1 * (nd - 1)) / nd
            c1 = tmp + (c1 * x * (2 * nd - 1)) / nd
    return c0 + c1 * x


def legval2d(x, y, c):
    return _evaluate(legval, c, x, y)


def legval3d(x, y, z, c):
    return _evaluate(legval, c, x, y, z)


def legvander(x, deg):
    ideg = operator.index(deg)
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    x = numpy.array(x, ndmin=1) + 0.0
    dims = (ideg + 1,) + x.shape
    dtyp = x.dtype
    v = numpy.empty(dims, dtype=dtyp)

    v[0] = x * 0 + 1
    if ideg > 0:
        v[1] = x
        for i in range(2, ideg + 1):
            v[i] = (v[i - 1] * x * (2 * i - 1) - v[i - 2] * (i - 1)) / i
    return numpy.moveaxis(v, 0, -1)


def legvander2d(x, y, deg):
    return _vander_nd_flat((legvander, legvander), (x, y), deg)


def legvander3d(x, y, z, deg):
    return _vander_nd_flat((legvander, legvander, legvander), (x, y, z), deg)


def legweight(x):
    return x * 0.0 + 1.0


def mapdomain(x, old, new):
    x = numpy.asanyarray(x)
    off, scl = mapparms(old, new)
    return off + scl * x


def mapparms(old, new):
    oldlen = old[1] - old[0]
    newlen = new[1] - new[0]
    off = (old[1] * new[0] - old[0] * new[1]) / oldlen
    scl = newlen / oldlen
    return off, scl


def poly2cheb(input):
    [input] = _as_series([input])

    output = 0

    for index in range(len(input) - 1, -1, -1):
        output = chebmulx(output)

        output = _add(output, input[index])

    return output


def poly2herm(input):
    [input] = _as_series([input])
    deg = len(input) - 1
    res = 0
    for i in range(deg, -1, -1):
        res = hermadd(hermmulx(res), input[i])
    return res


def poly2herme(input):
    [input] = _as_series([input])
    deg = len(input) - 1
    res = 0
    for i in range(deg, -1, -1):
        res = hermeadd(hermemulx(res), input[i])
    return res


def poly2lag(pol):
    [pol] = _as_series([pol])
    res = 0
    for p in pol[::-1]:
        res = lagadd(lagmulx(res), p)
    return res


def poly2leg(pol):
    [pol] = _as_series([pol])
    deg = len(pol) - 1
    res = 0
    for i in range(deg, -1, -1):
        res = legadd(legmulx(res), pol[i])
    return res


def polyadd(c1, c2):
    return _add(c1, c2)


def polycompanion(series):
    (series,) = _as_series([series])

    if len(series) < 2:
        raise ValueError

    if len(series) == 2:
        output = numpy.array([[-series[0] / series[1]]])
    else:
        n = series.shape[-1] - 1

        output = numpy.zeros([n, n], dtype=series.dtype)

        bot = numpy.reshape(output, -1)[n :: n + 1]

        bot[...] = 1

        output[:, -1] = output[:, -1] - (series[:-1] / series[-1])

    return output


def polyder(input, order: int = 1, scale: float = 1, axis: int = 0):
    output = numpy.array(input, ndmin=1)

    if output.dtype.char in "?bBhHiIlLqQpP":
        output = output + 0.0

    dtype = output.dtype

    cnt = operator.index(order)

    axis = operator.index(axis)

    if cnt < 0:
        raise ValueError

    axis = normalize_axis_index(axis, output.ndim)

    if cnt == 0:
        return output

    output = numpy.moveaxis(output, axis, 0)

    n = len(output)

    if cnt >= n:
        output = output[:1] * 0
    else:
        for _ in range(cnt):
            n = n - 1

            output = output * scale

            der = numpy.empty((n,) + output.shape[1:], dtype=dtype)

            for j in range(n, 0, -1):
                der[j - 1] = j * output[j]

            output = der

    output = numpy.moveaxis(output, 0, axis)

    return output


def polydiv(c1, c2):
    [c1, c2] = _as_series([c1, c2])
    if c2[-1] == 0:
        raise ZeroDivisionError()

    lc1 = len(c1)
    lc2 = len(c2)
    if lc1 < lc2:
        return c1[:1] * 0, c1
    elif lc2 == 1:
        return c1 / c2[-1], c1[:1] * 0
    else:
        dlen = lc1 - lc2
        scl = c2[-1]
        c2 = c2[:-1] / scl
        i = dlen
        j = lc1 - 1
        while i >= 0:
            c1[i:j] -= c2 * c1[j]
            i -= 1
            j -= 1
        return c1[j + 1 :] / scl, _trim_sequence(c1[: j + 1])


def polyfit(x, y, deg, rcond=None, full=False, w=None):
    return _fit(polyvander, x, y, deg, rcond, full, w)


def polyfromroots(roots):
    return _from_roots(polyline, polymul, roots)


def polygrid2d(x, y, c):
    return _grid(polyval, c, x, y)


def polygrid3d(x, y, z, c):
    return _grid(polyval, c, x, y, z)


def polyint(c, m=1, k=None, lbnd=0, scl=1, axis=0):
    if k is None:
        k = []
    c = numpy.array(c, ndmin=1)
    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c + 0.0
    cdt = c.dtype
    if not numpy.iterable(k):
        k = [k]
    cnt = operator.index(m)
    iaxis = operator.index(axis)
    if cnt < 0:
        raise ValueError("The order of integration must be non-negative")
    if len(k) > cnt:
        raise ValueError("Too many integration constants")
    if numpy.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if numpy.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")
    iaxis = normalize_axis_index(iaxis, c.ndim)

    if cnt == 0:
        return c

    k = list(k) + [0] * (cnt - len(k))
    c = numpy.moveaxis(c, iaxis, 0)
    for i in range(cnt):
        n = len(c)
        c *= scl
        if n == 1 and numpy.all(c[0] == 0):
            c[0] += k[i]
        else:
            tmp = numpy.empty((n + 1,) + c.shape[1:], dtype=cdt)
            tmp[0] = c[0] * 0
            tmp[1] = c[0]
            for j in range(1, n):
                tmp[j + 1] = c[j] / (j + 1)
            tmp[0] += k[i] - polyval(lbnd, tmp)
            c = tmp
    c = numpy.moveaxis(c, 0, iaxis)
    return c


def polyline(input, other):
    if other != 0:
        return numpy.array([input, other])
    else:
        return numpy.array([input])


def polymul(input, other):
    input, other = _as_series([input, other])

    output = numpy.convolve(input, other)

    output = _trim_sequence(output)

    return output


def polymulx(input):
    (input,) = _as_series([input])

    if len(input) == 1 and input[0] == 0:
        return input

    output = numpy.empty(len(input) + 1, dtype=input.dtype)

    output[0] = input[0] * 0.0

    output[1:] = input

    return output


def polypow(c, pow, maxpower=None):
    return _pow(numpy.convolve, c, pow, maxpower)


def polyroots(series):
    (series,) = _as_series([series])

    if len(series) < 2:
        return numpy.array([], dtype=series.dtype)

    if len(series) == 2:
        return numpy.array([-series[0] / series[1]])

    output = polycompanion(series)

    output = numpy.flip(output, axis=0)
    output = numpy.flip(output, axis=1)

    output = numpy.linalg.eigvals(output)

    output = numpy.sort(output)

    return output


def polysub(c1, c2):
    return _sub(c1, c2)


def polyval(x, c, tensor=True):
    c = numpy.array(c, ndmin=1)

    if c.dtype.char in "?bBhHiIlLqQpP":
        c = c + 0.0

    if isinstance(x, (tuple, list)):
        x = numpy.asarray(x)

    if isinstance(x, numpy.ndarray) and tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)

    c0 = c[-1] + x * 0

    for i in range(2, len(c) + 1):
        c0 = c[-i] + c0 * x

    return c0


def polyval2d(x, y, c):
    return _evaluate(polyval, c, x, y)


def polyval3d(x, y, z, c):
    return _evaluate(polyval, c, x, y, z)


def polyvalfromroots(x, output, tensor=True):
    output = numpy.array(output, ndmin=1)

    if output.dtype.char in "?bBhHiIlLqQpP":
        output = output.astype(numpy.float64)

    if isinstance(x, (tuple, list)):
        x = numpy.asarray(x)

    if isinstance(x, numpy.ndarray):
        if tensor:
            shape = (1,) * x.ndim

            output = numpy.reshape(output, [*output.shape, *shape])
        elif x.ndim >= output.ndim:
            raise ValueError

    return numpy.prod(x - output, axis=0)


def polyvander(input, degree):
    degree = operator.index(degree)

    if degree < 0:
        raise ValueError

    input = numpy.array(input, ndmin=1) + 0.0

    output = numpy.empty([degree + 1, *input.shape], dtype=input.dtype)

    output[0] = input * 0.0 + 1.0

    if degree > 0:
        output[1] = input

        for i in range(2, degree + 1):
            output[i] = output[i - 1] * input

    output = numpy.moveaxis(output, 0, -1)

    return output


def polyvander2d(x, y, deg):
    return _vander_nd_flat((polyvander, polyvander), (x, y), deg)


def polyvander3d(x, y, z, deg):
    return _vander_nd_flat((polyvander, polyvander, polyvander), (x, y, z), deg)


hermdomain = numpy.array([-1.0, 1.0])

hermedomain = numpy.array([-1.0, 1.0])

hermeone = numpy.array([1])

hermex = numpy.array([0, 1])

hermezero = numpy.array([0])

hermone = numpy.array([1])

hermx = numpy.array([0, 1 / 2])

hermzero = numpy.array([0])

lagdomain = numpy.array([0.0, 1.0])

lagone = numpy.array([1])

lagx = numpy.array([1, -1])

lagzero = numpy.array([0])

legdomain = numpy.array([-1.0, 1.0])

legone = numpy.array([1])

legx = numpy.array([0, 1])

legzero = numpy.array([0])

polydomain = numpy.array([-1.0, 1.0])

polyone = numpy.array([1])

polyx = numpy.array([0, 1])

polyzero = numpy.array([0])


__all__ = [
    "_div",
    "_pow",
    "_vander_nd",
    "_vander_nd_flat",
    "_as_series",
    "cheb2poly",
    "chebadd",
    "chebcompanion",
    "chebder",
    "chebdiv",
    "chebdomain",
    "chebfit",
    "chebfromroots",
    "chebgauss",
    "chebgrid2d",
    "chebgrid3d",
    "chebint",
    "chebinterpolate",
    "chebline",
    "chebmul",
    "chebmulx",
    "chebone",
    "chebpow",
    "chebpts1",
    "chebpts2",
    "chebroots",
    "chebsub",
    "chebtrim",
    "chebval",
    "chebval2d",
    "chebval3d",
    "chebvander",
    "chebvander2d",
    "chebvander3d",
    "chebweight",
    "chebx",
    "chebzero",
    "_get_domain",
    "herm2poly",
    "hermadd",
    "hermcompanion",
    "hermder",
    "hermdiv",
    "hermdomain",
    "herme2poly",
    "hermeadd",
    "hermecompanion",
    "hermeder",
    "hermediv",
    "hermedomain",
    "hermefit",
    "hermefromroots",
    "hermegauss",
    "hermegrid2d",
    "hermegrid3d",
    "hermeint",
    "hermeline",
    "hermemul",
    "hermemulx",
    "hermeone",
    "hermepow",
    "hermeroots",
    "hermesub",
    "hermetrim",
    "hermeval",
    "hermeval2d",
    "hermeval3d",
    "hermevander",
    "hermevander2d",
    "hermevander3d",
    "hermeweight",
    "hermex",
    "hermezero",
    "hermfit",
    "hermfromroots",
    "hermgauss",
    "hermgrid2d",
    "hermgrid3d",
    "hermint",
    "hermline",
    "hermmul",
    "hermmulx",
    "hermone",
    "hermpow",
    "hermroots",
    "hermsub",
    "hermtrim",
    "hermval",
    "hermval2d",
    "hermval3d",
    "hermvander",
    "hermvander2d",
    "hermvander3d",
    "hermweight",
    "hermx",
    "hermzero",
    "lag2poly",
    "lagadd",
    "lagcompanion",
    "lagder",
    "lagdiv",
    "lagdomain",
    "lagfit",
    "lagfromroots",
    "laggauss",
    "laggrid2d",
    "laggrid3d",
    "lagint",
    "lagline",
    "lagmul",
    "lagmulx",
    "lagone",
    "lagpow",
    "lagroots",
    "lagsub",
    "lagtrim",
    "lagval",
    "lagval2d",
    "lagval3d",
    "lagvander",
    "lagvander2d",
    "lagvander3d",
    "lagweight",
    "lagx",
    "lagzero",
    "leg2poly",
    "legadd",
    "legcompanion",
    "legder",
    "legdiv",
    "legdomain",
    "legfit",
    "legfromroots",
    "leggauss",
    "leggrid2d",
    "leggrid3d",
    "legint",
    "legline",
    "legmul",
    "legmulx",
    "legone",
    "legpow",
    "legroots",
    "legsub",
    "legtrim",
    "legval",
    "legval2d",
    "legval3d",
    "legvander",
    "legvander2d",
    "legvander3d",
    "legweight",
    "legx",
    "legzero",
    "mapdomain",
    "mapparms",
    "poly2cheb",
    "poly2herm",
    "poly2herme",
    "poly2lag",
    "poly2leg",
    "polyadd",
    "polycompanion",
    "polyder",
    "polydiv",
    "polydomain",
    "polyfit",
    "polyfromroots",
    "polygrid2d",
    "polygrid3d",
    "polyint",
    "polyline",
    "polymul",
    "polymulx",
    "polyone",
    "polypow",
    "polyroots",
    "polysub",
    "polytrim",
    "polyval",
    "polyval2d",
    "polyval3d",
    "polyvalfromroots",
    "polyvander",
    "polyvander2d",
    "polyvander3d",
    "polyx",
    "polyzero",
    "_trim_coefficients",
    "_trim_sequence",
]
