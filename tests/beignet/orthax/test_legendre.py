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
