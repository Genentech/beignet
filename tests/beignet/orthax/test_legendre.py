import beignet.orthax
import numpy
import numpy.testing

legcoefficients = [
    (numpy.array([1])),
    (numpy.array([0, 1])),
    (numpy.array([-1, 0, 3]) / 2),
    (numpy.array([0, -3, 0, 5]) / 2),
    (numpy.array([3, 0, -30, 0, 35]) / 8),
    (numpy.array([0, 15, 0, -70, 0, 63]) / 8),
    (numpy.array([-5, 0, 105, 0, -315, 0, 231]) / 16),
    (numpy.array([0, -35, 0, 315, 0, -693, 0, 429]) / 16),
    (numpy.array([35, 0, -1260, 0, 6930, 0, -12012, 0, 6435]) / 128),
    (numpy.array([0, 315, 0, -4620, 0, 18018, 0, -25740, 0, 12155]) / 128),
]


class TestEvaluation:
    c1d = numpy.array([2.0, 2.0, 2.0])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)
    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = numpy.polynomial.polynomial.polyval(x, [1.0, 2.0, 3.0])

    def test_legval(self):
        numpy.testing.assert_array_equal(beignet.orthax.legval([], [1]).size, 0)

        x = numpy.linspace(-1, 1)
        y = [numpy.polynomial.polynomial.polyval(x, c) for c in legcoefficients]
        for i in range(10):
            msg = f"At i={i}"
            tgt = y[i]
            res = beignet.orthax.legval(x, [0] * i + [1])
            numpy.testing.assert_array_almost_equal(res, tgt, err_msg=msg)

        for i in range(3):
            dims = [2] * i
            x = numpy.zeros(dims)
            numpy.testing.assert_array_equal(beignet.orthax.legval(x, [1]).shape, dims)
            numpy.testing.assert_array_equal(
                beignet.orthax.legval(x, [1, 0]).shape, dims
            )
            numpy.testing.assert_array_equal(
                beignet.orthax.legval(x, [1, 0, 0]).shape, dims
            )

    def test_legval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        numpy.testing.assert_raises(
            ValueError, beignet.orthax.legval2d, x1, x2[:2], self.c2d
        )

        tgt = y1 * y2
        res = beignet.orthax.legval2d(x1, x2, self.c2d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.legval2d(z, z, self.c2d)
        numpy.testing.assert_(res.shape == (2, 3))

    def test_legval3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        numpy.testing.assert_raises(
            ValueError, beignet.orthax.legval3d, x1, x2, x3[:2], self.c3d
        )

        tgt = y1 * y2 * y3
        res = beignet.orthax.legval3d(x1, x2, x3, self.c3d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.legval3d(z, z, z, self.c3d)
        numpy.testing.assert_(res.shape == (2, 3))

    def test_leggrid2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        tgt = numpy.einsum("i,j->ij", y1, y2)
        res = beignet.orthax.leggrid2d(x1, x2, self.c2d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.leggrid2d(z, z, self.c2d)
        numpy.testing.assert_(res.shape == (2, 3) * 2)

    def test_leggrid3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        tgt = numpy.einsum("i,j,k->ijk", y1, y2, y3)
        res = beignet.orthax.leggrid3d(x1, x2, x3, self.c3d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.leggrid3d(z, z, z, self.c3d)
        numpy.testing.assert_(res.shape == (2, 3) * 3)


class TestIntegral:
    def test_legint(self):  # noqa:C901
        numpy.testing.assert_raises(TypeError, beignet.orthax.legint, [0], 0.5)
        numpy.testing.assert_raises(ValueError, beignet.orthax.legint, [0], -1)
        numpy.testing.assert_raises(ValueError, beignet.orthax.legint, [0], 1, [0, 0])
        numpy.testing.assert_raises(ValueError, beignet.orthax.legint, [0], lbnd=[0])
        numpy.testing.assert_raises(ValueError, beignet.orthax.legint, [0], scl=[0])
        numpy.testing.assert_raises(TypeError, beignet.orthax.legint, [0], axis=0.5)

        for i in range(2, 5):
            k = [0] * (i - 2) + [1]
            res = beignet.orthax.legint([0], m=i, k=k)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.legtrim(res, tol=1e-6), [0, 1]
            )

        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [1 / scl]
            legpol = beignet.orthax.poly2leg(pol)
            legint = beignet.orthax.legint(legpol, m=1, k=[i])
            res = beignet.orthax.leg2poly(legint)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.legtrim(res, tol=1e-6),
                beignet.orthax.legtrim(tgt, tol=1e-6),
            )

        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            legpol = beignet.orthax.poly2leg(pol)
            legint = beignet.orthax.legint(legpol, m=1, k=[i], lbnd=-1)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.legval(-1, legint), i
            )

        for i in range(5):
            scl = i + 1
            pol = [0] * i + [1]
            tgt = [i] + [0] * i + [2 / scl]
            legpol = beignet.orthax.poly2leg(pol)
            legint = beignet.orthax.legint(legpol, m=1, k=[i], scl=2)
            res = beignet.orthax.leg2poly(legint)
            numpy.testing.assert_array_almost_equal(
                beignet.orthax.legtrim(res, tol=1e-6),
                beignet.orthax.legtrim(tgt, tol=1e-6),
            )

        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for _ in range(j):
                    tgt = beignet.orthax.legint(tgt, m=1)
                res = beignet.orthax.legint(pol, m=j)
                numpy.testing.assert_array_almost_equal(
                    beignet.orthax.legtrim(res, tol=1e-6),
                    beignet.orthax.legtrim(tgt, tol=1e-6),
                )

        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.orthax.legint(tgt, m=1, k=[k])
                res = beignet.orthax.legint(pol, m=j, k=list(range(j)))
                numpy.testing.assert_array_almost_equal(
                    beignet.orthax.legtrim(res, tol=1e-6),
                    beignet.orthax.legtrim(tgt, tol=1e-6),
                )

        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.orthax.legint(tgt, m=1, k=[k], lbnd=-1)
                res = beignet.orthax.legint(pol, m=j, k=list(range(j)), lbnd=-1)
                numpy.testing.assert_array_almost_equal(
                    beignet.orthax.legtrim(res, tol=1e-6),
                    beignet.orthax.legtrim(tgt, tol=1e-6),
                )

        for i in range(5):
            for j in range(2, 5):
                pol = [0] * i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = beignet.orthax.legint(tgt, m=1, k=[k], scl=2)
                res = beignet.orthax.legint(pol, m=j, k=list(range(j)), scl=2)
                numpy.testing.assert_array_almost_equal(
                    beignet.orthax.legtrim(res, tol=1e-6),
                    beignet.orthax.legtrim(tgt, tol=1e-6),
                )

    def test_legint_axis(self):
        c2d = numpy.random.random((3, 4))

        tgt = numpy.vstack([beignet.orthax.legint(c) for c in c2d.T]).T
        res = beignet.orthax.legint(c2d, axis=0)
        numpy.testing.assert_array_almost_equal(res, tgt)

        tgt = numpy.vstack([beignet.orthax.legint(c) for c in c2d])
        res = beignet.orthax.legint(c2d, axis=1)
        numpy.testing.assert_array_almost_equal(res, tgt)

        tgt = numpy.vstack([beignet.orthax.legint(c, k=3) for c in c2d])
        res = beignet.orthax.legint(c2d, k=3, axis=1)
        numpy.testing.assert_array_almost_equal(res, tgt)

    def test_legint_zerointord(self):
        numpy.testing.assert_array_equal(beignet.orthax.legint((1, 2, 3), 0), (1, 2, 3))


class TestDerivative:
    def test_legder(self):
        numpy.testing.assert_raises(TypeError, beignet.orthax.legder, [0], 0.5)
        numpy.testing.assert_raises(ValueError, beignet.orthax.legder, [0], -1)

        for i in range(5):
            tgt = [0] * i + [1]
            res = beignet.orthax.legder(tgt, m=0)
            numpy.testing.assert_array_equal(
                beignet.orthax.legtrim(res, tol=1e-6),
                beignet.orthax.legtrim(tgt, tol=1e-6),
            )

        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = beignet.orthax.legder(beignet.orthax.legint(tgt, m=j), m=j)
                numpy.testing.assert_array_almost_equal(
                    beignet.orthax.legtrim(res, tol=1e-6),
                    beignet.orthax.legtrim(tgt, tol=1e-6),
                )

        for i in range(5):
            for j in range(2, 5):
                tgt = [0] * i + [1]
                res = beignet.orthax.legder(
                    beignet.orthax.legint(tgt, m=j, scl=2), m=j, scl=0.5
                )
                numpy.testing.assert_array_almost_equal(
                    beignet.orthax.legtrim(res, tol=1e-6),
                    beignet.orthax.legtrim(tgt, tol=1e-6),
                )

    def test_legder_axis(self):
        c2d = numpy.random.random((3, 4))

        tgt = numpy.vstack([beignet.orthax.legder(c) for c in c2d.T]).T
        res = beignet.orthax.legder(c2d, axis=0)
        numpy.testing.assert_array_almost_equal(res, tgt)

        tgt = numpy.vstack([beignet.orthax.legder(c) for c in c2d])
        res = beignet.orthax.legder(c2d, axis=1)
        numpy.testing.assert_array_almost_equal(res, tgt)

    def test_legder_orderhigherthancoeff(self):
        c = (1, 2, 3, 4)
        numpy.testing.assert_array_equal(beignet.orthax.legder(c, 4), [0])
