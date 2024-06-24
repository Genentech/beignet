import beignet.orthax
import numpy
import numpy.testing

lagcoefficients = [
    (numpy.array([1]) / 1),
    (numpy.array([1, -1]) / 1),
    (numpy.array([2, -4, 1]) / 2),
    (numpy.array([6, -18, 9, -1]) / 6),
    (numpy.array([24, -96, 72, -16, 1]) / 24),
    (numpy.array([120, -600, 600, -200, 25, -1]) / 120),
    (numpy.array([720, -4320, 5400, -2400, 450, -36, 1]) / 720),
]


class TestEvaluation:
    c1d = numpy.array([9.0, -14.0, 6.0])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)
    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.orthax.polynomial.polyval(x, [1.0, 2.0, 3.0])

    def test_lagval(self):
        numpy.testing.assert_array_equal(beignet.orthax.lagval([], [1]).size, 0)

        x = numpy.linspace(-1, 1)
        y = [beignet.orthax.polynomial.polyval(x, c) for c in lagcoefficients]
        for i in range(7):
            msg = f"At i={i}"
            tgt = y[i]
            res = beignet.orthax.lagval(x, [0] * i + [1])
            numpy.testing.assert_array_almost_equal(res, tgt, err_msg=msg)

        for i in range(3):
            dims = [2] * i
            x = numpy.zeros(dims)
            numpy.testing.assert_array_equal(beignet.orthax.lagval(x, [1]).shape, dims)
            numpy.testing.assert_array_equal(
                beignet.orthax.lagval(x, [1, 0]).shape, dims
            )
            numpy.testing.assert_array_equal(
                beignet.orthax.lagval(x, [1, 0, 0]).shape, dims
            )

    def test_lagval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        numpy.testing.assert_raises(
            ValueError, beignet.orthax.lagval2d, x1, x2[:2], self.c2d
        )

        tgt = y1 * y2
        res = beignet.orthax.lagval2d(x1, x2, self.c2d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.lagval2d(z, z, self.c2d)
        numpy.testing.assert_(res.shape == (2, 3))

    def test_lagval3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        numpy.testing.assert_raises(
            ValueError, beignet.orthax.lagval3d, x1, x2, x3[:2], self.c3d
        )

        tgt = y1 * y2 * y3
        res = beignet.orthax.lagval3d(x1, x2, x3, self.c3d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.lagval3d(z, z, z, self.c3d)
        numpy.testing.assert_(res.shape == (2, 3))

    def test_laggrid2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        tgt = numpy.einsum("i,j->ij", y1, y2)
        res = beignet.orthax.laggrid2d(x1, x2, self.c2d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.laggrid2d(z, z, self.c2d)
        numpy.testing.assert_(res.shape == (2, 3) * 2)

    def test_laggrid3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        tgt = numpy.einsum("i,j,k->ijk", y1, y2, y3)
        res = beignet.orthax.laggrid3d(x1, x2, x3, self.c3d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.laggrid3d(z, z, z, self.c3d)
        numpy.testing.assert_(res.shape == (2, 3) * 3)
