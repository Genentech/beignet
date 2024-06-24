import beignet.orthax
import numpy
import numpy.testing

hermecoefficients = [
    (numpy.array([1])),
    (numpy.array([0, 1])),
    (numpy.array([-1, 0, 1])),
    (numpy.array([0, -3, 0, 1])),
    (numpy.array([3, 0, -6, 0, 1])),
    (numpy.array([0, 15, 0, -10, 0, 1])),
    (numpy.array([-15, 0, 45, 0, -15, 0, 1])),
    (numpy.array([0, -105, 0, 105, 0, -21, 0, 1])),
    (numpy.array([105, 0, -420, 0, 210, 0, -28, 0, 1])),
    (numpy.array([0, 945, 0, -1260, 0, 378, 0, -36, 0, 1])),
]


class TestEvaluation:
    c1d = numpy.array([4.0, 2.0, 3.0])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)
    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = numpy.polynomial.polynomial.polyval(x, [1.0, 2.0, 3.0])

    def test_hermeval(self):
        numpy.testing.assert_array_equal(beignet.orthax.hermeval([], [1]).size, 0)

        x = numpy.linspace(-1, 1)
        y = [numpy.polynomial.polynomial.polyval(x, c) for c in hermecoefficients]
        for i in range(10):
            msg = f"At i={i}"
            tgt = y[i]
            res = beignet.orthax.hermeval(x, [0] * i + [1])
            numpy.testing.assert_array_almost_equal(res, tgt, err_msg=msg)

        for i in range(3):
            dims = [2] * i
            x = numpy.zeros(dims)
            numpy.testing.assert_array_equal(
                beignet.orthax.hermeval(x, [1]).shape, dims
            )
            numpy.testing.assert_array_equal(
                beignet.orthax.hermeval(x, [1, 0]).shape, dims
            )
            numpy.testing.assert_array_equal(
                beignet.orthax.hermeval(x, [1, 0, 0]).shape, dims
            )

    def test_hermeval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        numpy.testing.assert_raises(
            ValueError, beignet.orthax.hermeval2d, x1, x2[:2], self.c2d
        )

        tgt = y1 * y2
        res = beignet.orthax.hermeval2d(x1, x2, self.c2d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.hermeval2d(z, z, self.c2d)
        numpy.testing.assert_(res.shape == (2, 3))

    def test_hermeval3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        numpy.testing.assert_raises(
            ValueError, beignet.orthax.hermeval3d, x1, x2, x3[:2], self.c3d
        )

        tgt = y1 * y2 * y3
        res = beignet.orthax.hermeval3d(x1, x2, x3, self.c3d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.hermeval3d(z, z, z, self.c3d)
        numpy.testing.assert_(res.shape == (2, 3))

    def test_hermegrid2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        tgt = numpy.einsum("i,j->ij", y1, y2)
        res = beignet.orthax.hermegrid2d(x1, x2, self.c2d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.hermegrid2d(z, z, self.c2d)
        numpy.testing.assert_(res.shape == (2, 3) * 2)

    def test_hermegrid3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        tgt = numpy.einsum("i,j,k->ijk", y1, y2, y3)
        res = beignet.orthax.hermegrid3d(x1, x2, x3, self.c3d)
        numpy.testing.assert_array_almost_equal(res, tgt)

        z = numpy.ones((2, 3))
        res = beignet.orthax.hermegrid3d(z, z, z, self.c3d)
        numpy.testing.assert_(res.shape == (2, 3) * 3)


class TestVander:
    x = numpy.random.random((3, 5)) * 2 - 1

    def test_hermevander(self):
        x = numpy.arange(3)
        v = beignet.orthax.hermevander(x, 3)
        numpy.testing.assert_(v.shape == (3, 4))
        for i in range(4):
            coef = [0] * i + [1]
            numpy.testing.assert_array_almost_equal(
                v[..., i], beignet.orthax.hermeval(x, coef)
            )

        x = numpy.array([[1, 2], [3, 4], [5, 6]])
        v = beignet.orthax.hermevander(x, 3)
        numpy.testing.assert_(v.shape == (3, 2, 4))
        for i in range(4):
            coef = [0] * i + [1]
            numpy.testing.assert_array_almost_equal(
                v[..., i], beignet.orthax.hermeval(x, coef)
            )

    def test_hermevander2d(self):
        x1, x2, x3 = self.x
        c = numpy.random.random((2, 3))
        van = beignet.orthax.hermevander2d(x1, x2, (1, 2))
        tgt = beignet.orthax.hermeval2d(x1, x2, c)
        res = numpy.dot(van, c.flat)
        numpy.testing.assert_array_almost_equal(res, tgt)

        van = beignet.orthax.hermevander2d([x1], [x2], (1, 2))
        numpy.testing.assert_(van.shape == (1, 5, 6))

    def test_hermevander3d(self):
        x1, x2, x3 = self.x
        c = numpy.random.random((2, 3, 4))
        van = beignet.orthax.hermevander3d(x1, x2, x3, (1, 2, 3))
        tgt = beignet.orthax.hermeval3d(x1, x2, x3, c)
        res = numpy.dot(van, c.flat)
        numpy.testing.assert_array_almost_equal(res, tgt)

        van = beignet.orthax.hermevander3d([x1], [x2], [x3], (1, 2, 3))
        numpy.testing.assert_(van.shape == (1, 5, 24))
