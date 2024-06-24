import beignet.orthax
import numpy
import numpy.testing

polycoefficients = [
    [1],
    [0, 1],
    [-1, 0, 2],
    [0, -3, 0, 4],
    [1, 0, -8, 0, 8],
    [0, 5, 0, -20, 0, 16],
    [-1, 0, 18, 0, -48, 0, 32],
    [0, -7, 0, 56, 0, -112, 0, 64],
    [1, 0, -32, 0, 160, 0, -256, 0, 128],
    [0, 9, 0, -120, 0, 432, 0, -576, 0, 256],
]


class TestEvaluation:
    c1d = numpy.array([1.0, 2.0, 3.0])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)
    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.orthax.polyval(x, [1.0, 2.0, 3.0])

    def test_polyval(self):
        numpy.testing.assert_equal(beignet.orthax.polyval([], [1]).size, 0)

        x = numpy.linspace(-1, 1)
        y = [x**i for i in range(5)]
        for i in range(5):
            tgt = y[i]
            res = beignet.orthax.polyval(x, [0] * i + [1])
            numpy.testing.assert_array_almost_equal(res, tgt)
        tgt = x * (x**2 - 1)
        res = beignet.orthax.polyval(x, [0, -1, 0, 1])
        numpy.testing.assert_array_almost_equal(res, tgt)

        for i in range(3):
            dims = [2] * i
            x = numpy.zeros(dims)
            numpy.testing.assert_equal(beignet.orthax.polyval(x, [1]).shape, dims)
            numpy.testing.assert_equal(beignet.orthax.polyval(x, [1, 0]).shape, dims)
            numpy.testing.assert_equal(beignet.orthax.polyval(x, [1, 0, 0]).shape, dims)

        mask = [False, True, False]
        mx = numpy.ma.array([1, 2, 3], mask=mask)
        res = numpy.polyval([7, 5, 3], mx)
        numpy.testing.assert_array_equal(res.mask, mask)

    def test_polyvalfromroots(self):
        numpy.testing.assert_raises(
            ValueError,
            beignet.orthax.polyvalfromroots,
            [1],
            [1],
            tensor=False,
        )

        numpy.testing.assert_equal(beignet.orthax.polyvalfromroots([], [1]).size, 0)
        numpy.testing.assert_(beignet.orthax.polyvalfromroots([], [1]).shape == (0,))

        numpy.testing.assert_equal(
            beignet.orthax.polyvalfromroots([], [[1] * 5]).size, 0
        )
        numpy.testing.assert_(
            beignet.orthax.polyvalfromroots([], [[1] * 5]).shape == (5, 0)
        )

        numpy.testing.assert_array_equal(beignet.orthax.polyvalfromroots(1, 1), 0)
        numpy.testing.assert_(
            beignet.orthax.polyvalfromroots(1, numpy.ones((3, 3))).shape == (3,)
        )

        x = numpy.linspace(-1, 1)
        y = [x**i for i in range(5)]
        for i in range(1, 5):
            tgt = y[i]
            res = beignet.orthax.polyvalfromroots(x, [0] * i)
            numpy.testing.assert_array_almost_equal(res, tgt)
        tgt = x * (x - 1) * (x + 1)
        res = beignet.orthax.polyvalfromroots(x, [-1, 0, 1])
        numpy.testing.assert_array_almost_equal(res, tgt)

        for i in range(3):
            dims = [2] * i
            x = numpy.zeros(dims)
            numpy.testing.assert_equal(
                beignet.orthax.polyvalfromroots(x, [1]).shape, dims
            )
            numpy.testing.assert_equal(
                beignet.orthax.polyvalfromroots(x, [1, 0]).shape, dims
            )
            numpy.testing.assert_equal(
                beignet.orthax.polyvalfromroots(x, [1, 0, 0]).shape, dims
            )

        ptest = [15, 2, -16, -2, 1]
        r = beignet.orthax.polyroots(ptest)
        x = numpy.linspace(-1, 1)
        numpy.testing.assert_array_almost_equal(
            beignet.orthax.polyval(x, ptest),
            beignet.orthax.polyvalfromroots(x, r),
        )

        rshape = (3, 5)
        x = numpy.arange(-3, 2)
        r = numpy.random.randint(-5, 5, size=rshape)
        res = beignet.orthax.polyvalfromroots(x, r, tensor=False)
        tgt = numpy.empty(r.shape[1:])
        for ii in range(tgt.size):
            tgt[ii] = beignet.orthax.polyvalfromroots(x[ii], r[:, ii])
        numpy.testing.assert_array_equal(res, tgt)

        x = numpy.vstack([x, 2 * x])
        res = beignet.orthax.polyvalfromroots(x, r, tensor=True)
        tgt = numpy.empty(r.shape[1:] + x.shape)
        for ii in range(r.shape[1]):
            for jj in range(x.shape[0]):
                tgt[ii, jj, :] = beignet.orthax.polyvalfromroots(x[jj], r[:, ii])
        numpy.testing.assert_array_equal(res, tgt)

    def test_polyval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        numpy.testing.assert_raises_regex(
            ValueError,
            "incompatible",
            beignet.orthax.polyval2d,
            x1,
            x2[:2],
            self.c2d,
        )

        tgt = y1 * y2
        res = beignet.orthax.polyval2d(x1, x2, self.c2d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.polyval2d(z, z, self.c2d)
        numpy.testing.assert_(res.shape == (2, 3))

    def test_polyval3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        numpy.testing.assert_raises_regex(
            ValueError,
            "incompatible",
            beignet.orthax.polyval3d,
            x1,
            x2,
            x3[:2],
            self.c3d,
        )

        tgt = y1 * y2 * y3
        res = beignet.orthax.polyval3d(x1, x2, x3, self.c3d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.polyval3d(z, z, z, self.c3d)
        numpy.testing.assert_(res.shape == (2, 3))

    def test_polygrid2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        tgt = numpy.einsum("i,j->ij", y1, y2)
        res = beignet.orthax.polygrid2d(x1, x2, self.c2d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.polygrid2d(z, z, self.c2d)
        numpy.testing.assert_(res.shape == (2, 3) * 2)

    def test_polygrid3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        tgt = numpy.einsum("i,j,k->ijk", y1, y2, y3)
        res = beignet.orthax.polygrid3d(x1, x2, x3, self.c3d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.polygrid3d(z, z, z, self.c3d)
        numpy.testing.assert_(res.shape == (2, 3) * 3)
